#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#include<math.h>
#include<dirent.h>
#include"lib/graph.h"
#include"lib/rules.h"

#include"cuda.h"
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include"lib/cuda_common.cuh"


void generate_teps(Graph *G, int steps, char* file_name, bool* init_state);

__global__ void evolve_tep_gpu(bool* TEP, rules* bRules, int counterB, rules* sRules, float* density, 
double* resolution, int* degree, int* adjList, int* indexes, int* sum_of_states,
int rules_size, int number_of_nodes, int steps);

__global__ void shannon_entropy(int* sum_of_states, int steps, int number_of_nodes, int counterB,
int* histogram, int attributes);

__global__ void word_entropy_histogram(bool* TEP, int steps, int number_of_nodes, int counterB, int rules_size, int attributes, int* histogram);

__global__ void population(bool* TEP, int steps, int number_of_nodes, int counterB, int rules_size, int attributes, float* population);

__global__ void tp_correlation(bool* TEP, int steps, int number_of_nodes, int counterB, int rules_size, int attributes, float* correlation);

bool* createRandomInitState(int number_of_nodes);

arr_adjList* generate_gpu_adjlist(Graph* G);

int main(void){

    int steps=350;

    struct dirent *dir;
    const char *dir_path = "data/rulesel/";
    DIR *d;
    d = opendir(dir_path);
    int counter =1;
    int MAX_NODES = 2000;

    bool* init_state = createRandomInitState(MAX_NODES);

    FILE *chkp = fopen("data/checkpoint.csv", "r");
    int check;
    fscanf(chkp, "%d", &check);
    fclose(chkp);
    if(d){
        while((dir = readdir(d)) != NULL){
            if(!strcmp(get_filename_ext(dir->d_name), "txt")){
                if(counter >= check){
                    printf("Iniciando grafo %d: %s\n", counter, dir->d_name);
                
                    char *file_path = (char*) malloc(sizeof(char)*50);
                    strcpy(file_path,dir_path);
                    strcat(file_path, dir->d_name);

                    Graph* G = read_adjList(file_path);

                    generate_teps(G, steps, dir->d_name, init_state);

            
                    free(file_path);
                    free(G);
                }
                
                counter++;
                chkp = fopen("data/checkpoint.csv", "w");
                fprintf(chkp, "%d", counter); 
                fclose(chkp);

            }
            
        }
    }
    free(init_state);
    closedir(d);

    return 0;
}


bool* createRandomInitState(int number_of_nodes){
    bool* init_state = (bool*) malloc(sizeof(bool)*number_of_nodes);

    for(int i=0;i<number_of_nodes;i++){
        init_state[i] = rand()&1;
    }

    return init_state;
}

void generate_teps(Graph *G, int steps, char* file_name, bool* init_state){

    //in this function, we will use a number of threads equivalent to the number of survive rules
    //for example, for 8 neighbors, there will be 512 survive rules. Since a graph has n nodes and each node 
    //is counted as a thread, there will be 512*n threads. In this implementation, we will set as 512 the 
    //number of threads in each block and the number of blocks in each grid as n.

    //get size of rules
    int rules_size = 1;
    for(int i=0;i<NB_SIZE+1;i++){
        rules_size*=2;
    }

    //set block size as rules size and grid size as number of vertices
    dim3 block(rules_size);
    dim3 grid(G->numVertices);

    //get number of nodes
    int number_of_nodes = G->numVertices;

    //create birth and survive rule structure
    rules* bRules = getAllRules();
    rules* sRules = getAllRules();

    
    //allocate memory for gpu
    rules* gpu_bRules;
    rules* gpu_sRules;
    gpuErrchk(cudaMalloc((void**)&gpu_bRules, sizeof(rules)*rules_size));
    gpuErrchk(cudaMalloc((void**)&gpu_sRules, sizeof(rules)*rules_size));

    gpuErrchk(cudaMemcpy(gpu_bRules, bRules, sizeof(rules)*rules_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpu_sRules, sRules, sizeof(rules)*rules_size, cudaMemcpyHostToDevice));

    //free memory for host
    free(bRules);
    free(sRules);


    //resolution of densities to calculate the rules
    double* resolution = (double*)malloc(sizeof(double) * (NB_SIZE + 2));

    for (int i = 0; i < NB_SIZE + 2; i++) {
        resolution[i] = (i) / (double)(NB_SIZE + 1);
    }

    //transfer resolution value to device
    double* gpu_resolution;
    gpuErrchk(cudaMalloc((void**)&gpu_resolution, sizeof(double)*(NB_SIZE+2)));
    gpuErrchk(cudaMemcpy(gpu_resolution, resolution, sizeof(double)*(NB_SIZE+2), cudaMemcpyHostToDevice));

    free(resolution);

    //calculate initial density for all nodes
    int* alive_neighbors = (int*) malloc(sizeof(int)*number_of_nodes);
    int* degree = (int*)malloc(sizeof(int)*number_of_nodes);
    float* density = (float*)malloc(sizeof(float)*number_of_nodes*rules_size);
    
    for(int i=0;i<G->numVertices;i++){
        alive_neighbors[i]=0;
        node *p = G->adjLists[i];
        degree[i]=0;
        while(p){
            if(init_state[p->vertex] == 1){
                alive_neighbors[i]+=1;
            }
            degree[i]++;
            p = p->next;
        }
    }

    for(int i=0;i<G->numVertices;i++){
        if(alive_neighbors[i]==0){
            for(int j=0;j<rules_size; j++){
                density[i+j*number_of_nodes]=0;
            }
            
        }
        else{
            for(int j=0;j<rules_size; j++){
                density[i+j*number_of_nodes] = (float)alive_neighbors[i]/(float)degree[i];
            }
        }
    }

    free(alive_neighbors);

    //since we are working with (number_of_nodes*number_of_survive_rules) number of threads, in order to reduce
    //the overhead we will reduce the number of copies of the TEP. We will only make one copy of the TEP at
    //every birth rule iteration. In order to do this, first we must allocate memory from the host, so that
    //we can copy the information from device to host.
    bool* TEP = (bool*) malloc(sizeof(bool)*number_of_nodes*steps*rules_size);
    
    //create array in device with same size as the host
    bool* gpu_TEP;
    
    gpuErrchk(cudaMalloc((void**)&gpu_TEP, sizeof(bool)*number_of_nodes*steps*rules_size));
    
    //copy the initial state array to the TEP structure at the device, respecting the indexes
    for(int i=0;i<rules_size;i++){
        gpuErrchk(cudaMemcpy(&gpu_TEP[i*G->numVertices*steps], &init_state[0], 
        sizeof(bool)*number_of_nodes, cudaMemcpyHostToDevice));
    }

    //also, create arrays for both degree and density. Since density is dinamically updated, we will extend
    //the array to be the block size. Note that there is no need to do this for the degree array, since
    //it is a static array
    int* gpu_degree;
    float* gpu_density;
    
    gpuErrchk(cudaMalloc((void**)&gpu_degree, sizeof(int)*number_of_nodes));
    gpuErrchk(cudaMalloc((void**)&gpu_density, sizeof(float)*number_of_nodes*rules_size));

    //copy host values to device arrays
    gpuErrchk(cudaMemcpy(gpu_degree, degree, sizeof(int)*number_of_nodes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpu_density, density, sizeof(float)*number_of_nodes, cudaMemcpyHostToDevice)); 
    //free memory for host arrays
    free(degree);

    //create two array structure on host representing the adjacencies from the graph
    int* adjList = (int*) malloc(sizeof(int)*2*G->numTransitions);
    int* indexes = (int*) malloc(sizeof(int)*G->numVertices);

    int counter=0;
    for(int i=0;i<G->numVertices;i++){
        indexes[i] = counter;
        node*p = G->adjLists[i];
        while(p){
            adjList[counter] = p->vertex;
            counter++;
            p = p->next;
        }
    }

    //and copy it to the device
    int* gpu_adjList;
    int* gpu_indexes;
    gpuErrchk(cudaMalloc((void**)&gpu_adjList, sizeof(int)*2*G->numTransitions));
    gpuErrchk(cudaMalloc((void**)&gpu_indexes, sizeof(int)*G->numVertices));
    gpuErrchk(cudaMemcpy(gpu_adjList, adjList, sizeof(int)*2*G->numTransitions, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpu_indexes, indexes, sizeof(int)*G->numVertices, cudaMemcpyHostToDevice));

    //free memory from host
    free(adjList);
    free(indexes);

    //create array to sum each cell state, in order to calculate shannon entropy
    int* gpu_sum_of_states;
    gpuErrchk(cudaMalloc((void**)&gpu_sum_of_states, sizeof(int)*G->numVertices*rules_size));

    //define number of attributes in shannon entropy histogram
    int attributes = 20;
    
    //declare variable for shannon entropy histogram for boths host and device
    int* cpu_shannon_histogram = (int*)malloc(sizeof(int)*rules_size*rules_size*attributes);

    int* gpu_shannon_histogram;
    gpuErrchk(cudaMalloc((void**)&gpu_shannon_histogram, sizeof(int)*rules_size*rules_size*attributes));
    gpuErrchk(cudaMemset(gpu_shannon_histogram, 0, sizeof(int)*rules_size*rules_size*attributes));

    //declare variable for word entropy for both host and device
    int* cpu_word_histogram = (int*) malloc(sizeof(int)*rules_size*rules_size*attributes);

    int* gpu_word_histogram;
    gpuErrchk(cudaMalloc((void**)&gpu_word_histogram, sizeof(int)*rules_size*rules_size*attributes));
    gpuErrchk(cudaMemset(gpu_word_histogram, 0, sizeof(int)*rules_size*rules_size*attributes));

    //declare variable to calculate population on both host and device
    float* cpu_population = (float*) malloc(sizeof(float)*attributes*rules_size*rules_size);
    float* gpu_population;
    gpuErrchk(cudaMalloc((void**)&gpu_population, sizeof(float)*attributes*rules_size*rules_size));
    gpuErrchk(cudaMemset(gpu_population, 0, sizeof(float)*attributes*rules_size*rules_size));

    //declare variable to calculate two point correlation on both host and device
    float* cpu_correlation = (float*)malloc(sizeof(float)*attributes*rules_size*rules_size);
    float* gpu_correlation;
    gpuErrchk(cudaMalloc((void**)&gpu_correlation, sizeof(float)*attributes*rules_size*rules_size));
    gpuErrchk(cudaMemset(gpu_correlation, 0, sizeof(float)*attributes*rules_size*rules_size));

    //loop for B rules from 0 to 2^(Number of neighbors)
    for(int counterB=0;counterB<512;counterB++){
        //printf("\nBirth Rule Number: %d\n", counterB);
        
        evolve_tep_gpu <<<block, grid>>> (gpu_TEP, gpu_bRules, counterB, gpu_sRules, gpu_density, gpu_resolution,
        gpu_degree, gpu_adjList, gpu_indexes, gpu_sum_of_states, rules_size, G->numVertices, steps);
        gpuErrchk(cudaMemcpy(gpu_density, density, sizeof(float)*number_of_nodes*rules_size, cudaMemcpyHostToDevice));

        shannon_entropy <<<block, grid>>> (gpu_sum_of_states, steps, G->numVertices, counterB,
        gpu_shannon_histogram, attributes);
        gpuErrchk(cudaDeviceSynchronize());

        population <<<block, steps>>> (gpu_TEP, steps, G->numVertices, counterB, rules_size, attributes, gpu_population);
        gpuErrchk(cudaDeviceSynchronize());

        word_entropy_histogram <<<block, grid>>> (gpu_TEP, steps, G->numVertices, counterB, rules_size, attributes, gpu_word_histogram);
        gpuErrchk(cudaDeviceSynchronize());

        tp_correlation <<<block, steps>>> (gpu_TEP, steps, G->numVertices, counterB, rules_size, attributes, gpu_correlation);
        gpuErrchk(cudaDeviceSynchronize());

    }

    //transfer shannon entropy histogram from device to host
    gpuErrchk(cudaMemcpy(cpu_shannon_histogram, gpu_shannon_histogram, 
    sizeof(int)*rules_size*rules_size*attributes, cudaMemcpyDeviceToHost));

    //transfer word entropy histogram from device to host
    gpuErrchk(cudaMemcpy(cpu_word_histogram, gpu_word_histogram, 
    sizeof(int)*rules_size*rules_size*attributes, cudaMemcpyDeviceToHost));

    //transfer population array from device to host
    gpuErrchk(cudaMemcpy(cpu_population, gpu_population, sizeof(float)*rules_size*rules_size*attributes, cudaMemcpyDeviceToHost));

    //transfer correlation array from device to host
    gpuErrchk(cudaMemcpy(cpu_correlation, gpu_correlation, sizeof(float)*rules_size*rules_size*attributes, cudaMemcpyDeviceToHost));

    //write file with results of shannon entropy
    FILE *fout_shannon;
    char* path = "data/measures/shannon/";
    char* pout = (char*) malloc(sizeof(char)*(50));
    char* file_wo_ext = (char*) malloc(sizeof(char)*(50));
    char* ext = ".csv";
    snprintf(file_wo_ext, strlen(file_name)-3, "%s", file_name);
    strcat(file_wo_ext, ext);
    strcpy(pout, path);
    strcat(pout, file_wo_ext);

    fout_shannon = fopen(pout, "w");
    printf("Save name: %s\n", pout);
    for(int i=0;i<rules_size*rules_size*attributes;i++){
        if((i+1)%attributes==0){
            fprintf(fout_shannon, "%d\n", cpu_shannon_histogram[i]);
        }
        else{
            fprintf(fout_shannon, "%d,", cpu_shannon_histogram[i]);
        }
    }
    
    fclose(fout_shannon);

    //write file with results of word entropy
    FILE *fout_word;
    path = "data/measures/word/";
    pout = (char*) malloc(sizeof(char)*(50));
    file_wo_ext = (char*) malloc(sizeof(char)*(50));
    ext = ".csv";
    snprintf(file_wo_ext, strlen(file_name)-3, "%s", file_name);
    strcat(file_wo_ext, ext);
    strcpy(pout, path);
    strcat(pout, file_wo_ext);

    fout_word = fopen(pout, "w");
    printf("Save name: %s\n", pout);
    for(int i=0;i<rules_size*rules_size*attributes;i++){
        if((i+1)%attributes==0){
            fprintf(fout_word, "%d\n", cpu_word_histogram[i]);
        }
        else{
            fprintf(fout_word, "%d,", cpu_word_histogram[i]);
        }
    }
    
    fclose(fout_word);

    //write file with results of population
    FILE *fout_pop;
    path = "data/measures/population/";
    pout = (char*) malloc(sizeof(char)*(50));
    file_wo_ext = (char*) malloc(sizeof(char)*(50));
    ext = ".csv";
    snprintf(file_wo_ext, strlen(file_name)-3, "%s", file_name);
    strcat(file_wo_ext, ext);
    strcpy(pout, path);
    strcat(pout, file_wo_ext);

    fout_pop = fopen(pout, "w");
    printf("Save name: %s\n", pout);
    for(int i=0;i<rules_size*rules_size*attributes;i++){
        if((i+1)%attributes==0){
            fprintf(fout_pop, "%.4f\n", cpu_population[i]);
        }
        else{
            fprintf(fout_pop, "%.4f,", cpu_population[i]);
        }
    }
    
    fclose(fout_pop);

    //write file with results of population
    FILE *fout_cor;
    path = "data/measures/correlation/";
    pout = (char*) malloc(sizeof(char)*(50));
    file_wo_ext = (char*) malloc(sizeof(char)*(50));
    ext = ".csv";
    snprintf(file_wo_ext, strlen(file_name)-3, "%s", file_name);
    strcat(file_wo_ext, ext);
    strcpy(pout, path);
    strcat(pout, file_wo_ext);

    fout_cor = fopen(pout, "w");
    printf("Save name: %s\n", pout);
    for(int i=0;i<rules_size*rules_size*attributes;i++){
        if((i+1)%attributes==0){
            fprintf(fout_pop, "%.4f\n", cpu_correlation[i]);
        }
        else{
            fprintf(fout_pop, "%.4f,", cpu_correlation[i]);
        }
    }
    
    fclose(fout_cor);


    //free remaining allocated memory from host and device
    //free(pout);
    //free(file_wo_ext);
    free(TEP);
    free(cpu_shannon_histogram);
    cudaFree(gpu_sum_of_states);
    cudaFree(gpu_shannon_histogram);
    cudaFree(gpu_TEP);
    cudaFree(gpu_degree);
    cudaFree(gpu_density);
    free(density);
    cudaFree(gpu_resolution);
    cudaFree(gpu_bRules);
    cudaFree(gpu_sRules);
    cudaFree(gpu_adjList);
    cudaFree(gpu_indexes);
    free(cpu_population);
    free(cpu_correlation);
    cudaFree(gpu_correlation);
    cudaFree(gpu_population);
    free(cpu_word_histogram);
    cudaFree(gpu_word_histogram);
}


__global__ void evolve_tep_gpu(bool* TEP, rules* bRules, int counterB, rules* sRules, float* density, 
    double* resolution, int* degree, int* adjList, int* indexes, int* sum_of_states,  int rules_size, 
    int number_of_nodes, int steps){

    int gid = blockDim.x*blockIdx.x + threadIdx.x;
    int rule_number = gid/number_of_nodes;
    int node_number = gid % number_of_nodes;
    int rule_offset = number_of_nodes*steps;

    sum_of_states[gid] = 0;

    for(int i=1;i<steps;i++){
        TEP[node_number+i*number_of_nodes+rule_number*rule_offset] = false;      
        //birth rule dominates
        if(TEP[node_number+(i-1)*number_of_nodes+rule_number*rule_offset]==false){
            for(int k=0;k < NB_SIZE+1; k++){
                if(bRules[counterB].rule[k] == true &&
                density[node_number + number_of_nodes*rule_number] >= resolution[k] &&
                density[node_number + number_of_nodes*rule_number] < resolution[k+1]){
                    TEP[rule_number*rule_offset + node_number+ i*number_of_nodes] = true;
                    sum_of_states[node_number+number_of_nodes*rule_number] += 1;
                    for(int d=0; d<degree[node_number]; d++){
                        atomicAdd(&density[adjList[indexes[node_number]+d]+number_of_nodes*rule_number], 1/(float)degree[adjList[indexes[node_number]+d]]);
                    }
                    break;
                }
            }
        } 
        else{
            for(int k=0;k < NB_SIZE+1; k++){
                if(sRules[rule_number].rule[k] == true &&
                density[node_number + number_of_nodes*rule_number] >= resolution[k] &&
                density[node_number + number_of_nodes*rule_number] < resolution[k+1]){
                    TEP[rule_number*rule_offset + node_number+ i*number_of_nodes] = true;
                    sum_of_states[node_number+number_of_nodes*rule_number] += 1;
                    break;
                }
            }
            if(TEP[node_number+i*number_of_nodes+rule_number*rule_offset] == false){
                for(int d=0; d<degree[node_number]; d++){
                    atomicAdd(&density[adjList[indexes[node_number]+d]+number_of_nodes*rule_number], -1/(float)degree[adjList[indexes[node_number]+d]]);
                }
            }
        }
        __syncthreads();
    }
}


__global__ void shannon_entropy(int* sum_of_states, int steps, int number_of_nodes, int counterB,
    int* histogram, int attributes){
    
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    int rules_size=512;
    int size_of_batch = rules_size*attributes;
    int offset = gid / number_of_nodes;
    float division = (float)1/(float)attributes + 0.00001;
    
    if(gid < rules_size*number_of_nodes){
        int sum = sum_of_states[gid];
        float p1 = (float) sum/(float)steps;
        float p0 = (float)(steps-sum) / (float)steps;
        float value = (-1) * (p1 * log2f(p1+0.001) + p0 * log2f(p0+0.001));
        int index = (int) (value/division);
        atomicAdd(&histogram[index+attributes*offset+counterB*size_of_batch],1);
        __syncthreads();
    }
}

//requires rules_size*steps to calculate. To avoid concurrency, every thread will calculate the population
// a timestep
__global__ void population(bool* TEP, int steps, int number_of_nodes, int counterB, int rules_size, int attributes, float* population){
    //population -> size attributes*rules_size

    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    int rule_offset = rules_size*counterB*attributes;
    int resolution = steps/attributes;
    int step_idx = gid % steps;
    int rule_idx = gid / (number_of_nodes*steps);

    

    if(gid < attributes*rules_size){
        population[gid+rule_offset] = 0; 
        for(int i=0;i<number_of_nodes;i++){
            population[gid+rule_offset]+=(float)TEP[step_idx*resolution*number_of_nodes+steps*number_of_nodes*rule_idx+i]/(float)number_of_nodes;
        }
        __syncthreads();
    }
    
}

// calculate word entropy histogram, this histogram has 20 indexes. Larger word length considered is 40 and every index in the
// histogram has 2 bins.
__global__ void word_entropy_histogram(bool* TEP, int steps, int number_of_nodes, int counterB, int rules_size, int attributes,  int* histogram){
   
    int gid = blockIdx.x*blockDim.x+threadIdx.x;

    int size_of_batch = rules_size*attributes;
    int rule_offset = rules_size*counterB*attributes;
    int node_idx = gid % number_of_nodes;
    int rule_idx = gid / number_of_nodes;

    if(gid < rules_size*number_of_nodes){
        int word_size=0;
        for(int i=0;i<steps;i++){
            if(TEP[node_idx + i*number_of_nodes + rule_idx*number_of_nodes*steps] == false){
                if(word_size > 0 && word_size <=40){
                    int index = word_size/2.001;
                    atomicAdd(&histogram[index+rule_idx*attributes+rule_offset], 1);
                }
                word_size=0;
            }
            else{
                word_size+=1;
            }
        }
        __syncthreads();
    }
    

}

//pick the sum_of_states array to make the two point correlation. We pick values for r between 1 and 20, we paralelize it for each thread to calculate every step
__global__ void tp_correlation(bool* TEP, int steps, int number_of_nodes, int counterB, int rules_size, int attributes, float* correlation){
    
    int gid = blockIdx.x*blockDim.x + threadIdx.x;

    int size_of_batch = rules_size*counterB*attributes;
    int step_idx = gid % steps;
    int rule_idx = gid / steps;

    if(gid < rules_size*steps){
        for(int r = 1; r<= attributes;r++){
            int counter=0;
            int prod = 0;
            int sum_m = 0;
            int sum_r = 0;
            for(int i=0;i<number_of_nodes-r;i++){
                int value_m;
                int value_r;
                if(TEP[i + step_idx*number_of_nodes + rule_idx*number_of_nodes*steps]){
                    value_m = 1;
                }
                else{
                    value_m = -1;
                }
                if(TEP[i+r + step_idx*number_of_nodes + rule_idx*number_of_nodes*steps]){
                    value_r = 1;
                }
                else{
                    value_r = -1;
                }

                sum_m+=value_m;
                sum_r+=value_r;
                prod+=value_m*value_r;
                counter++;
            }
            float mean_prod = (float) prod/(float)counter;
            float mean_m = (float) sum_m/(float)counter;
            float mean_r = (float)sum_r/(float)counter;

            float value = (mean_prod - mean_m*mean_r)/steps;

            atomicAdd(&correlation[r-1 + rule_idx*attributes + size_of_batch], value);    
        }     
        __syncthreads();  
    }
    
}