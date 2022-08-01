#include"rules.h"
#include<stdio.h>
#include<stdlib.h>
#include"filesave.h"
#include"statistics.cuh"


__device__ __forceinline__ float atomicMinFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
            __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

__device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

//evolve TEP for full set of rules, in this case there is a counter to the index of birth rules
//in the parameters
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
                density[node_number + number_of_nodes*rule_number] <= resolution[k+1]){
                    TEP[rule_number*rule_offset + node_number+ i*number_of_nodes] = true;
                    sum_of_states[node_number+number_of_nodes*rule_number] += 1;
                    for(int d=0; d<degree[node_number]; d++){
                        atomicAdd(&density[adjList[indexes[node_number]+d]+number_of_nodes*rule_number], 1/(float)degree[adjList[indexes[node_number]+d]]);
                        if(density[adjList[indexes[node_number]+d]+number_of_nodes*rule_number]>1){
                            density[adjList[indexes[node_number]+d]+number_of_nodes*rule_number] = 1;
                        }
                    }
                    break;
                }
            }
        }
        else{
            for(int k=0;k < NB_SIZE+1; k++){
                if(sRules[rule_number].rule[k] == true &&
                density[node_number + number_of_nodes*rule_number] >= resolution[k] &&
                density[node_number + number_of_nodes*rule_number] <= resolution[k+1]){
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

// evolve both TEP and density TEP for a specific number of steps using selected rules
__global__ void evolve_tep_dtep_gpu_selected(bool* TEP, float* dtep, lifelike* rules, float* density, 
    double* resolution, int* degree, int* adjList, int* indexes, int rules_size, 
    int number_of_nodes, int steps){

    int gid = blockIdx.x*blockDim.x+threadIdx.x;
    int rule_offset = number_of_nodes*steps;
    int rule_number = blockIdx.x;
    int iterations = number_of_nodes/(blockDim.x+0.0001) +1;
    
    for(int i=1;i<steps;i++){
        for(int iter=0;iter<iterations;iter++){
            int node_number = blockDim.x*iter + threadIdx.x;
            if(node_number < number_of_nodes){
                __syncthreads();
                float dt = density[node_number+number_of_nodes*rule_number];
                __syncthreads();
                TEP[node_number+i*number_of_nodes+rule_number*rule_offset] = false;      
                //birth rule dominates
                if(TEP[node_number+(i-1)*number_of_nodes+rule_number*rule_offset]==false){
                    for(int k=0;k < NB_SIZE+1; k++){
                        if(rules[rule_number].bRule[k] == true &&
                        dt >= resolution[k] &&
                        dt <= resolution[k+1]){
                            TEP[rule_number*rule_offset + node_number+ i*number_of_nodes] = true;
                            for(int d=0; d<degree[node_number]; d++){
                                if(degree[adjList[indexes[node_number]+d]]!= 0){
                                   atomicAdd(&density[adjList[indexes[node_number]+d]+number_of_nodes*rule_number], 1/(float)degree[adjList[indexes[node_number]+d]]);
                                }
                            }
                            break;
                        }
                    }
                }
                else{
                    for(int k=0;k < NB_SIZE+1; k++){
                        if(rules[rule_number].sRule[k] == true &&
                        dt >= resolution[k] &&
                        dt <= resolution[k+1]){
                            TEP[rule_number*rule_offset + node_number+ i*number_of_nodes] = true;
                            break;
                        }
                    }
                    if(TEP[node_number+i*number_of_nodes+rule_number*rule_offset] == false){
                        for(int d=0; d<degree[node_number]; d++){
                            if(degree[adjList[indexes[node_number]+d]]!= 0){
                                atomicAdd(&density[adjList[indexes[node_number]+d]+number_of_nodes*rule_number], -1/(float)degree[adjList[indexes[node_number]+d]]);
                            }
                        }
                    }
                }
                __syncthreads();
                dtep[node_number+number_of_nodes*i+rule_number*number_of_nodes*steps] = density[node_number+number_of_nodes*rule_number];
                __syncthreads();
            }
        }
    }
}


bool* createRandomInitState(int number_of_nodes){
    bool* init_state = (bool*) malloc(sizeof(bool)*number_of_nodes);

    for(int i=0;i<number_of_nodes;i++){
        init_state[i] = rand()&1;
    }

    return init_state;
}

//generate density teps according to a selected set of rules
void generate_dteps_selected_statistics(Graph *G, int steps, char* file_name, bool* init_state, const char* rules_path, const char* output_path){
    //prepare selected rules
    int rules_size;
    lifelike* rules = getSelectedRules(rules_path, &rules_size);
    lifelike* gpu_Rules;
    gpuErrchk(cudaMalloc((void**)&gpu_Rules, sizeof(lifelike)*rules_size));
    gpuErrchk(cudaMemcpy(gpu_Rules,rules, sizeof(lifelike)*rules_size, cudaMemcpyHostToDevice));

    //prepare cuda threads
    int maxThreadsPerBlock = 1024;
    int blocksPerRule = G->numVertices/maxThreadsPerBlock + 1;

    int block_size = min(G->numVertices, maxThreadsPerBlock);

    dim3 block(block_size);
    dim3 grid(rules_size);

    //resolution of densities to calculate the rules
    double* resolution = (double*)malloc(sizeof(double) * (NB_SIZE + 2));

    for (int i = 0; i < NB_SIZE + 2; i++) {
        resolution[i] = (i) / (double)(NB_SIZE + 1);
    }

    //transfer resolution value to device
    double* gpu_resolution;
    gpuErrchk(cudaMalloc((void**)&gpu_resolution, sizeof(double)*(NB_SIZE+2)));
    gpuErrchk(cudaMemcpy(gpu_resolution, resolution, sizeof(double)*(NB_SIZE+2), cudaMemcpyHostToDevice));

    //calculate initial density for all nodes
    int* alive_neighbors = (int*) malloc(sizeof(int)*G->numVertices);
    int* degree = (int*)malloc(sizeof(int)*G->numVertices);
    float* density = (float*)malloc(sizeof(float)*G->numVertices);
    
    for(int i=0;i<G->numVertices;i++){
        alive_neighbors[i]=0;
        node *p = G->adjLists[i];
        degree[i]=0;
        while(p != NULL){
            if(init_state[p->vertex] == 1){
                alive_neighbors[i]+=1;
            }
            degree[i]++;
            p = p->next;
        }
         
    }

    for(int i=0;i<G->numVertices;i++){
        if(alive_neighbors[i]==0){
            density[i]=0;
            
        }
        else{
            if(degree[i] != 0){
                density[i] = (float)alive_neighbors[i]/(float)degree[i];
            }
            else{
                density[i] = 0;
            }
        }
    }

    // //also, create arrays for both degree and density. Since density is dinamically updated, we will extend
    // //the array to be the block size. Note that there is no need to do this for the degree array, since
    // //it is a static array
    int* gpu_degree;
    float* gpu_density;
    
    gpuErrchk(cudaMalloc((void**)&gpu_degree, sizeof(int)*G->numVertices));
    gpuErrchk(cudaMalloc((void**)&gpu_density, sizeof(float)*G->numVertices*rules_size));

    // //copy host values to device arrays
    gpuErrchk(cudaMemcpy(gpu_degree, degree, sizeof(int)*G->numVertices, cudaMemcpyHostToDevice));
    for(int i=0;i<rules_size;i++){
        gpuErrchk(cudaMemcpy(&gpu_density[G->numVertices*i], density, sizeof(float)*G->numVertices, cudaMemcpyHostToDevice)); 
    }
    
    //create two array structure on host representing the adjacencies from the graph
    int* adjList = (int*) malloc(sizeof(int)*G->numTransitions);
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

    // // //and copy it to the device
    int* gpu_adjList;
    int* gpu_indexes;
    gpuErrchk(cudaMalloc((void**)&gpu_adjList, sizeof(int)*G->numTransitions));
    gpuErrchk(cudaMalloc((void**)&gpu_indexes, sizeof(int)*G->numVertices));
    gpuErrchk(cudaMemcpy(gpu_adjList, adjList, sizeof(int)*G->numTransitions, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpu_indexes, indexes, sizeof(int)*G->numVertices, cudaMemcpyHostToDevice));


    // //since we are working with (number_of_nodes*number_of_rules) number of threads, in order to reduce
    // //the overhead we will reduce the number of copies of the TEP. We will only make one copy of the TEP at
    // //every birth rule iteration. In order to do this, first we must allocate memory from the host, so that
    // //we can copy the information from device to host.
    bool* TEP = (bool*) malloc(sizeof(bool)*G->numVertices*steps*rules_size);
    float* dtep = (float*) malloc(sizeof(float)*G->numVertices*steps*rules_size);
    
    // //create array in device with same size as the host
    bool* gpu_TEP;
    float* gpu_dtep; 
    gpuErrchk(cudaMalloc((void**)&gpu_TEP, sizeof(bool)*G->numVertices*steps*rules_size));
    gpuErrchk(cudaMalloc((void**)&gpu_dtep, sizeof(float)*G->numVertices*steps*rules_size));


    // //copy the initial state array to the TEP structure at the device, respecting the indexes
    for(int i=0;i<rules_size;i++){
        gpuErrchk(cudaMemcpy(&gpu_TEP[i*G->numVertices*steps], &init_state[0], 
        sizeof(bool)*G->numVertices, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(&gpu_dtep[i*G->numVertices*steps], &density[0],
        sizeof(float)*G->numVertices, cudaMemcpyHostToDevice));
    }
        
    evolve_tep_dtep_gpu_selected <<<grid, block>>> (gpu_TEP, gpu_dtep, gpu_Rules,
                                gpu_density, gpu_resolution, gpu_degree, gpu_adjList, 
                                gpu_indexes, rules_size, G->numVertices, steps);
    gpuErrchk(cudaDeviceSynchronize());

     //transfer TEP from device to host
    gpuErrchk(cudaMemcpy(TEP, gpu_TEP, 
    sizeof(bool)*rules_size*G->numVertices*steps, cudaMemcpyDeviceToHost));

    // // //transfer word entropy histogram from device to host
    gpuErrchk(cudaMemcpy(dtep, gpu_dtep, 
    sizeof(float)*rules_size*G->numVertices*steps, cudaMemcpyDeviceToHost));

    int transient_offset=20;
    int attributes=20;

    //  // //declare variable for shannon entropy histogram for boths host and device
    // int* cpu_shannon_normal = (int*)malloc(sizeof(int)*rules_size*attributes);
    // int* gpu_shannon_normal;
    // gpuErrchk(cudaMalloc((void**)&gpu_shannon_normal, sizeof(int)*rules_size*attributes));
    // gpuErrchk(cudaMemset(gpu_shannon_normal, 0, sizeof(int)*rules_size*attributes));

    // // // //declare variable for word entropy for both host and device
    // int* cpu_word_normal = (int*) malloc(sizeof(int)*rules_size*attributes);
    // int* gpu_word_normal;
    // gpuErrchk(cudaMalloc((void**)&gpu_word_normal, sizeof(int)*rules_size*attributes));
    // gpuErrchk(cudaMemset(gpu_word_normal, 0, sizeof(int)*rules_size*attributes));

    // int* lz_tep = (int*) malloc(sizeof(int)*G->numVertices*rules_size);
    // int* gpu_lz_tep;
    // gpuErrchk(cudaMalloc((void**)&gpu_lz_tep, sizeof(int)*G->numVertices*rules_size));

    // shannon_entropy <<<grid, block>>> (gpu_TEP, steps, G->numVertices, 0,
    // gpu_shannon_normal, attributes, rules_size, transient_offset);
    // gpuErrchk(cudaDeviceSynchronize());

    // word_entropy_histogram <<<grid, block>>> (gpu_TEP, steps, G->numVertices, 0, rules_size, 
    // attributes, 40, gpu_word_normal, transient_offset);
    // gpuErrchk(cudaDeviceSynchronize());

    // lz_complexity_cpu(TEP, steps, G->numVertices, rules_size, lz_tep, transient_offset);

    // // //transfer shannon entropy histogram from device to host
    // gpuErrchk(cudaMemcpy(cpu_shannon_normal, gpu_shannon_normal, 
    // sizeof(int)*rules_size*attributes, cudaMemcpyDeviceToHost));

    // // //transfer word entropy histogram from device to host
    // gpuErrchk(cudaMemcpy(cpu_word_normal, gpu_word_normal, 
    // sizeof(int)*rules_size*attributes, cudaMemcpyDeviceToHost));


    // // // //transfer lempel ziv array from device to host
    // gpuErrchk(cudaMemcpy(lz_tep, gpu_lz_tep, 
    // sizeof(int)*rules_size*G->numVertices, cudaMemcpyDeviceToHost));


    // //computes and saves the density measures for different bin sizes
    int n_histograms = 8;
    int density_histogram_size[n_histograms] = {20,40,60,80,100,120,140,160};
     //global density histogram
    int* cpu_density_histogram = (int*)malloc(sizeof(int)*rules_size*(density_histogram_size[n_histograms-1]+1)*n_histograms);
    int* gpu_density_histogram;
    gpuErrchk(cudaMalloc((void**)&gpu_density_histogram, sizeof(int)*rules_size*(density_histogram_size[n_histograms-1]+1)*n_histograms));
    gpuErrchk(cudaMemset(gpu_density_histogram, 0, sizeof(int)*rules_size*(density_histogram_size[n_histograms-1]+1)*n_histograms));
    int* cpu_sdensity_histogram = (int*)malloc(sizeof(int)*rules_size*(density_histogram_size[n_histograms-1]+1)*n_histograms);
    int* gpu_sdensity_histogram;
    gpuErrchk(cudaMalloc((void**)&gpu_sdensity_histogram, sizeof(int)*rules_size*(density_histogram_size[n_histograms-1]+1)*n_histograms));
    gpuErrchk(cudaMemset(gpu_sdensity_histogram, 0, sizeof(int)*rules_size*(density_histogram_size[n_histograms-1]+1)*n_histograms));


    for(int i=0;i<n_histograms;i++){
        density_histogram <<<grid, block>>> (gpu_dtep, steps, G->numVertices, 0, rules_size,
        &gpu_density_histogram[(density_histogram_size[n_histograms-1]+1)*rules_size*i], 
        density_histogram_size[i], transient_offset, density_histogram_size[n_histograms-1]);
        gpuErrchk(cudaDeviceSynchronize());
        density_state_histogram <<<grid, block>>> (gpu_dtep, gpu_TEP, steps, G->numVertices, 0, rules_size,
        &gpu_sdensity_histogram[(density_histogram_size[n_histograms-1]+1)*rules_size*i], 
        density_histogram_size[i], transient_offset, density_histogram_size[n_histograms-1]);
        gpuErrchk(cudaDeviceSynchronize());
    }

    gpuErrchk(cudaMemcpy(cpu_density_histogram, gpu_density_histogram, 
    sizeof(int)*rules_size*(density_histogram_size[n_histograms-1]+1)*n_histograms, 
    cudaMemcpyDeviceToHost));

    gpuErrchk(cudaMemcpy(cpu_sdensity_histogram, gpu_sdensity_histogram, 
    sizeof(int)*rules_size*(density_histogram_size[n_histograms-1]+1)*n_histograms, 
    cudaMemcpyDeviceToHost));
   
    
    for(int i=0;i<rules_size;i++){
        char* file_wo_ext = (char*) malloc(sizeof(char)*100);
        char* str = (char*) malloc(sizeof(char)*100);
        snprintf(file_wo_ext, strlen(file_name)-3, "%s", file_name);
        sprintf(str,"_net_rule_%d_density",i+1);
        strcat(file_wo_ext, str);
        strcat(file_wo_ext, ".csv");

        save_csv_dTEPs(file_wo_ext, output_path, &dtep[G->numVertices*steps*i], degree, G->numVertices, steps, transient_offset);
        

        free(file_wo_ext);
        free(str);
    }

    for(int i=0;i<rules_size;i++){
        char* file_wo_ext = (char*) malloc(sizeof(char)*100);
        char* str = (char*) malloc(sizeof(char)*100);
        snprintf(file_wo_ext, strlen(file_name)-3, "%s", file_name);
        sprintf(str,"_net_rule_%d_binary",i+1);
        strcat(file_wo_ext, str);
        strcat(file_wo_ext, ".csv");

        save_csv_TEPs(file_wo_ext, output_path, &TEP[G->numVertices*steps*i], degree, G->numVertices, steps, transient_offset);
        
        
        free(file_wo_ext);
        free(str);
    }

    char* file_wo_ext = (char*) malloc(sizeof(char)*100);
    char* str = "_net_degree";
    snprintf(file_wo_ext, strlen(file_name)-3, "%s", file_name);
    strcat(file_wo_ext, str);
    strcat(file_wo_ext, ".csv");

    save_csv_int(file_wo_ext, output_path, degree, G->numVertices, 1);

    str = "_net_density_histogram";
    snprintf(file_wo_ext, strlen(file_name)-3, "%s", file_name);
    strcat(file_wo_ext, str);
    strcat(file_wo_ext, ".csv");

    save_csv_int(file_wo_ext, output_path, cpu_density_histogram, density_histogram_size[n_histograms-1]+1, rules_size*n_histograms);

    str = "_net_density_state_histogram";
    snprintf(file_wo_ext, strlen(file_name)-3, "%s", file_name);
    strcat(file_wo_ext, str);
    strcat(file_wo_ext, ".csv");

    save_csv_int(file_wo_ext, output_path, cpu_sdensity_histogram, density_histogram_size[n_histograms-1]+1, rules_size*n_histograms);


    // str = "_net_shannon_tep";
    // snprintf(file_wo_ext, strlen(file_name)-3, "%s", file_name);
    // strcat(file_wo_ext, str);
    // strcat(file_wo_ext, ".csv");

    // save_csv_int(file_wo_ext, output_path, cpu_shannon_normal, attributes, rules_size);


    // str = "_net_word_tep";
    // snprintf(file_wo_ext, strlen(file_name)-3, "%s", file_name);
    // strcat(file_wo_ext, str);
    // strcat(file_wo_ext, ".csv");

    // save_csv_int(file_wo_ext, output_path, cpu_word_normal, attributes, rules_size);


    // str = "_net_lz_tep";
    // snprintf(file_wo_ext, strlen(file_name)-3, "%s", file_name);
    // strcat(file_wo_ext, str);
    // strcat(file_wo_ext, ".csv");

    // save_csv_int(file_wo_ext, output_path, lz_tep, G->numVertices, rules_size);

    free(file_wo_ext);


    cudaFree(gpu_degree);
    free(degree);
    cudaFree(gpu_density);
    free(density);
    cudaFree(gpu_resolution);
    free(resolution);
    free(alive_neighbors);
    
    free(rules);
    cudaFree(gpu_Rules);

    free(adjList);
    free(indexes);
    cudaFree(gpu_adjList);
    cudaFree(gpu_indexes);
    

    free(TEP);
    free(dtep);
    cudaFree(gpu_TEP);
    cudaFree(gpu_dtep);
    
    // free(cpu_shannon_normal);
    // cudaFree(gpu_shannon_normal);
    // free(cpu_word_normal);
    // cudaFree(gpu_word_normal);
    // cudaFree(gpu_lz_tep);
    // free(lz_tep);

    free(cpu_density_histogram);
    cudaFree(gpu_density_histogram);
    free(cpu_sdensity_histogram);
    cudaFree(gpu_sdensity_histogram);
}