/**
* @details
* Algorithmic optimisations allowed: calculating the outdegrees, buffer swap,
* change the storage format.
**/
 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
 
#define GRAPH_ORDER 1000
#define GRAPH_SIZE 100000
#define DAMPING_FACTOR 0.85
#define MAX_TIME 10
 
double adjacency_matrix[GRAPH_ORDER][GRAPH_ORDER];
 
void initialize_graph() {
    for (int i = 0; i < GRAPH_ORDER; i++) {
        for (int j = 0; j < GRAPH_ORDER; j++) {
            adjacency_matrix[i][j] = 0.0;
        }
    }
}
 
void add_edge(int source, int destination) {
    if (source >= 0 && source < GRAPH_ORDER && destination >= 0 && destination < GRAPH_ORDER) {
        adjacency_matrix[source][destination] = 1.0;
    }
}

void calculate_pagerank(double pagerank[]) {
    double initial_rank = 1.0 / GRAPH_ORDER;
 
    for (int i = 0; i < GRAPH_ORDER; i++) {
        pagerank[i] = initial_rank;
    }
 
    double damping_value = (1.0 - DAMPING_FACTOR) / GRAPH_ORDER;
 
    double diff = 1.0;
    size_t iteration = 0;
    double start = omp_get_wtime();
    double elapsed = omp_get_wtime() - start;
    double time_per_iteration = 0;
    double new_pagerank[GRAPH_ORDER];
    for (int i = 0; i < GRAPH_ORDER; i++) {
        new_pagerank[i] = 0.0;
    }
    while(elapsed < MAX_TIME && (elapsed + time_per_iteration) < MAX_TIME) {
        double iteration_start = omp_get_wtime();
 
        for (int i = 0; i < GRAPH_ORDER; i++) {
            new_pagerank[i] = 0.0;
        }
 
		for (int i = 0; i < GRAPH_ORDER; i++) {
			for (int j = 0; j < GRAPH_ORDER; j++) {
				if (adjacency_matrix[j][i] == 1.0) {
					int outdegree = 0;
				 
					for (int k = 0; k < GRAPH_ORDER; k++) {
						if (adjacency_matrix[j][k] == 1.0) {
							outdegree++;
						}
					}
					new_pagerank[i] += pagerank[j] / (double)outdegree;
				}
			}
		}
 
        for (int i = 0; i < GRAPH_ORDER; i++) {
            new_pagerank[i] = DAMPING_FACTOR * new_pagerank[i] + damping_value;
        }
 
        diff = 0.0;
        for (int i = 0; i < GRAPH_ORDER; i++) {
            diff += fabs(new_pagerank[i] - pagerank[i]);
        }
 
        for (int i = 0; i < GRAPH_ORDER; i++) {
            pagerank[i] = new_pagerank[i];
        }
 
		double iteration_end = omp_get_wtime();
		elapsed = omp_get_wtime() - start;
		iteration++;
		time_per_iteration = elapsed / iteration;
    }
    printf("%zu iterations achieved in %.2f seconds\n", iteration, elapsed);
}

int main() {
    double start = omp_get_wtime();
    initialize_graph();
 
    srand(123);
    for(int i = 0; i < GRAPH_SIZE; i++)
    {
        add_edge(rand() % GRAPH_ORDER, rand() % GRAPH_ORDER);
    }
 
    double pagerank[GRAPH_ORDER];
    calculate_pagerank(pagerank);
 
    double sum_ranks = 0.0;
    for(int i = 0; i < GRAPH_ORDER; i++) {
        if(i % 100 == 0)
        {
            printf("PageRank of Node %d: %.4lf\n", i, pagerank[i]);
        }
        sum_ranks += pagerank[i];
    }
    printf("Total rank = %f.\n", sum_ranks);
    double end = omp_get_wtime();
 
    printf("Time taken: %.2f seconds.\n", end - start);
 
    return 0;
}
