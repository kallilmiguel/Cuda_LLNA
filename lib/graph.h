#include<stdio.h>
#include<stdlib.h>
#include<dirent.h>
#include<string.h>
#include<math.h>

typedef struct node{
    int vertex;
    struct node* next;
}node;

typedef struct{
    int numVertices;
    int numTransitions;
    node** adjLists;
}Graph;

typedef struct{
    int* neighbors;
    int* indexes;
}arr_adjList;

node* createNode(int v);
Graph* createGraph(int vertices);
void addEdge(Graph* graph, int source, int destiny);
const char *get_filename_ext(const char *filename);
Graph* read_adjList(char* path);
void printGraph(Graph* G, int n_of_iterations);
arr_adjList* convert_adjList_to_array(Graph* G);


//Create a node
node* createNode(int v){
    node *newNode = (node*) malloc(sizeof(node));
    newNode->vertex = v;
    newNode->next = NULL;
    return newNode;
}

//Create a graph
Graph* createGraph(int vertices){
    Graph *graph = (Graph*) malloc(sizeof(Graph));
    graph->numVertices = vertices;
    graph->numTransitions = 0;

    graph->adjLists = (node**) malloc(vertices * sizeof(node*));

    int i;
    for(i=0; i<vertices;i++){
        graph->adjLists[i] = NULL;
    }

    return graph;
}

//Add edge (in bidirectional graph)
void addEdge(Graph* graph, int source, int destiny){
    //add edge from s to d
    node* newNode = createNode(destiny);
    newNode->next = graph->adjLists[source];
    graph->adjLists[source] = newNode;
    
    //add edge from d to s
    node* newNode2 = createNode(source);
    newNode2->next = graph->adjLists[destiny];
    graph->adjLists[destiny] = newNode2;

    graph->numTransitions++;
    
}

//print the graph
void printGraph(Graph* graph, int n_of_iterations)
{
    int v;
    for (v = 0; v < n_of_iterations; v++) {
        node* temp = graph->adjLists[v];
        printf("\nVertex %d\n ", v);
        while (temp) {
            printf("%d -> ", temp->vertex);
            temp = temp->next;
        }
        printf("\n");
    }
}

Graph* read_adjList(char* path){

    
    FILE *f =fopen(path, "r");

    int counter = 0;
    int lines_until_vertex_read=0;

    while(1){
        char c = fgetc(f);
        if(c == EOF){
            break;
        }
        else if(c == '\n'){
            counter++;
        }
        else if(c == '#'){
            while(c != '\n' && c != EOF){
                c = fgetc(f);
            }
        }
    }

    Graph *G = createGraph(counter-lines_until_vertex_read);

    rewind(f);

    char* str = (char*) malloc(sizeof(char)*10000);

    for(int i=0;i<lines_until_vertex_read;i++){
        fgets(str, 10000, f);
    }

    while(fgets(str,10000, f)!= NULL){
        int vertex;
        if(str[0] != '#'){
            sscanf(str,"%d", &vertex);
            int edge;
            int char_offset;
            int size = strlen(str);
            if(vertex >1){
                char_offset = 1+ceil(log10((double)vertex));
            }
            else{
                char_offset = 2;
            }
            while(sscanf(&str[char_offset], "%d \n", &edge)!= NULL && char_offset< size){
                if(edge > 1){
                    char_offset += 1+ceil(log10((double)edge)); 
                }
                else{
                    char_offset +=2;
                }
                addEdge(G, vertex, edge);
            }
        }
    }

    fclose(f);
    free(str);
    return G;
}

arr_adjList* convert_adjList_to_array(Graph* G){

    int number_of_nodes = G->numVertices;

    int transition_counter=0;
    
    for(int i=0;i<number_of_nodes;i++){
        node *p = G->adjLists[i];
        while(p){
            transition_counter++;
            p = p->next;
        }
    }

    arr_adjList* adjList;
    adjList = (arr_adjList*) malloc(sizeof(arr_adjList));

    adjList->indexes = (int*)malloc(sizeof(int)*number_of_nodes);
    adjList->neighbors = (int*)malloc(sizeof(int)*transition_counter);

    for(int i=0; i<transition_counter;i++){
        node *p = G->adjLists[i];
        while(p){
            adjList->neighbors[i] = p->vertex;
            p = p->next;
            i++;
        }
    }

    return adjList;
}

const char *get_filename_ext(const char *filename){
    const char *dot = strrchr(filename, '.');
    if(!dot || dot == filename) return "";
    return dot + 1;
}
