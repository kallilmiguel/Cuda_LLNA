#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#include<math.h>
#include<dirent.h>
#include"lib/graph.h"


#include"cuda.h"
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include"lib/cuda_common.cuh"
#include"lib/tep.cuh"

int main(int argc, char* argv[]){

    struct dirent *dir;
    const char* data_path = argv[1];
    const char* rule_path = argv[2];
    const char* output_path = argv[3];
    const char* seed_path = argv[4];

    DIR *d;
    d = opendir(data_path);
    int counter =1;
    int MAX_NODES = 5000;
    int steps=350;

    FILE* fseed = fopen(seed_path, "r");

    bool* init_state = (bool*) malloc(sizeof(bool)*MAX_NODES);
    for(int i=0;i<MAX_NODES;i++){
        fscanf(fseed, "%d\n", &init_state[i]);
    }
    
    //Graph* G = (Graph*) malloc(sizeof(Graph));
    if(d){
        while((dir = readdir(d)) != NULL){
            if(!strcmp(get_filename_ext(dir->d_name), "txt")){
                printf("Iniciando grafo %d: %s\n", counter, dir->d_name);
            
                char *file_path = (char*) malloc(sizeof(char)*100);
                strcpy(file_path,data_path);
                strcat(file_path, dir->d_name);
                
                Graph* G = read_adjList(file_path);
                
                printf("Nodes: %d, edges: %d\n", G->numVertices, G->numTransitions);
                //generate_dteps_selected(G, steps, dir->d_name, init_state, rule_path, output_path);
                generate_dteps_selected_statistics(G, steps, dir->d_name, init_state, rule_path, output_path);
        
                free(file_path);
                //freeGraph(G);
                free(G);

                counter++;

            }
            
        }
    }
    //free(init_state);
    closedir(d);
    //fclose(fseed);

    return 0;
}
