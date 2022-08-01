#include<stdio.h>
#include<stdlib.h>
#include<dirent.h>
#include"lib/graph.h"
#include"lib/rules.h"

#define NB_SIZE 8

void generate_teps(Graph *G, int steps);
bool* evolve_single_tep(Graph *G, rules bRules, rules sRules,bool* init_state, double* density, 
    double* resolution, int* degree, int steps);

int main(void){
    int steps=350;

    struct dirent *dir;
    char *dir_path = "data/watts/";
    DIR *d;
    d = opendir(dir_path);

    if(d){
        while((dir = readdir(d)) != NULL){
            if(!strcmp(get_filename_ext(dir->d_name), "txt")){
                printf("Iniciando grafo: %s\n", dir->d_name);
                
                
                char *file_path = (char*) malloc(sizeof(char)*(strlen(dir_path)+strlen(dir->d_name)));
                strcpy(file_path,dir_path);
                strcat(file_path, dir->d_name);

                Graph* G = read_adjList(file_path);
                
                //generate_teps(G, steps);

                free(file_path);
                free(G);

            }
            
        }
    }
    closedir(d);

    return 0;
}


void generate_teps(Graph *G, int steps){

    //create birth and survive rule structure
    rules* bRules = getAllRules();
    rules* sRules = getAllRules();

    //resolution of densities to calculate the rules
    double* resolution = (double*)malloc(sizeof(double) * (NB_SIZE + 2));

    for (int i = 0; i < NB_SIZE + 2; i++) {
        resolution[i] = (i) / (double)(NB_SIZE + 1);
    }

    //determine initial state for all TEPs
    bool* init_state = (bool*) malloc(sizeof(bool)*G->numVertices);

    for(int i=0;i<G->numVertices;i++){
        init_state[i] = rand()&1;
    }

    //calculate initial density for all nodes
    int* alive_neighbors = (int*) malloc(sizeof(int)*G->numVertices);
    int* degree = (int*)malloc(sizeof(int)*G->numVertices);
    double* density = (double*)malloc(sizeof(double)*G->numVertices);
    
    for(int i=0;i<G->numVertices;i++){
        alive_neighbors[i]=0;
        node *p = G->adjLists[i];
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
            density[i]=0;
        }
        else{
            density[i] = (double)alive_neighbors[i]/(double)degree[i];
        }
    }

    free(alive_neighbors);
    
    //create counter for both B and S rules
    int counterB;
    int counterS;

    //loop for both rules from 0 to 2^(Number of neighbors)
    for(counterB=0;counterB<512;counterB++){
        printf("\nBirth Rule Number: %d\n", counterB);
        for(counterS=0;counterS<512;counterS++){

            bool* TEP = evolve_single_tep(G, bRules[counterB], sRules[counterS], init_state, 
            density, resolution, degree, steps);
        }
    }
}

bool* evolve_single_tep(Graph *G, rules bRules, rules sRules, bool* init_state, double* density, 
    double* resolution, int* degree, int steps){
     
    bool* TEP = (bool*)malloc(sizeof(bool)*G->numVertices*steps);
    for(int i=0;i<G->numVertices;i++){
        TEP[i] = init_state[i];
    }
    
    for(int i=0;i<G->numVertices;i++){
        for(int j=1;j<steps;j++){
            TEP[i+j*G->numVertices] = false;
            for(int k=0;k<NB_SIZE+1;k++){            
                //birth rule dominates
                if(TEP[i+ (j-1)*G->numVertices]==false && 
                bRules.rule[k]==true && 
                density[i] >= resolution[k] &&
                density[i] < resolution[k+1]){
                    TEP[i+j*G->numVertices] = true;
                    node *p = G->adjLists[i];
                    
                    while(p){
                        density[p->vertex]+=1/degree[p->vertex];
                        p = p->next;
                    }
                    break;
                }
                //survive rule dominates
                else if(TEP[i+ (j-1)*G->numVertices]==true && 
                sRules.rule[k]==true && 
                density[i] >= resolution[k] &&
                density[i] < resolution[k+1]){
                    TEP[i+j*G->numVertices] = true;
                    break;
                }
            }
            if(TEP[i+(j-1)*G->numVertices]==true && TEP[i+j*G->numVertices] == false){
                node *p = G->adjLists[i];
                while(p){
                    density[p->vertex]-=1/degree[p->vertex];
                    p = p->next;
                }
            }
        }
    }

    return TEP;
}