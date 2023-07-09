/**
 * @file main.c
 * @brief This file provides you with the original implementation of pagerank.
 * Your challenge is to optimise it using OpenMP and/or MPI.
 * @details Algorithmic optimisations allowed: calculating the outdegrees,
 * buffer swap, change the storage format.
 * @author Ludovic Capelli (l.capelli@epcc.ed.ac.uk)
**/
 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

/// The number of vertices in the graph.
#define GRAPH_ORDER 1000
/// The number of edges in the graph.
#define GRAPH_SIZE 100000
/// Parameters used in pagerank convergence, do not change.
#define DAMPING_FACTOR 0.85
/// The number of seconds to not exceed forthe calculation loop.
#define MAX_TIME 10

/**
 * @brief Indicates which vertices are connected.
 * @details If an edge links vertex A to vertex B, then adjacency_matrix[A][B]
 * will be 1.0. The absence of edge is represented with value 0.0.
 * Redundant edges are still represented with value 1.0.
 */
double adjacency_matrix[GRAPH_ORDER][GRAPH_ORDER];
 
void initialize_graph(void)
{
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        for(int j = 0; j < GRAPH_ORDER; j++)
        {
            adjacency_matrix[i][j] = 0.0;
        }
    }
}

/**
 * @brief Adds a directed edge to the graph, from \p source to \p destination.
 * @details If the edge already exists, it does nothing.
 * @param source The identifier of the vertex at the start of the edge.
 * @param destination The identifier of the vertex at the end of the edge.
 */
void add_edge(int source, int destination)
{
    if (source >= 0 && source < GRAPH_ORDER && destination >= 0 && destination < GRAPH_ORDER)
    {
        adjacency_matrix[source][destination] = 1.0;
    }
}

/**
 * @brief Calculates the pagerank of all vertices in the graph.
 * @param pagerank The array in which store the final pageranks.
 */
void calculate_pagerank(double pagerank[])
{
    double initial_rank = 1.0 / GRAPH_ORDER;
 
    // Initialise all vertices to 1/n.
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        pagerank[i] = initial_rank;
    }
 
    double damping_value = (1.0 - DAMPING_FACTOR) / GRAPH_ORDER;
 
    double diff = 1.0;
    size_t iteration = 0;
    double start = omp_get_wtime();
    double elapsed = omp_get_wtime() - start;
    double time_per_iteration = 0;
    double new_pagerank[GRAPH_ORDER];
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        new_pagerank[i] = 0.0;
    }

    // If we exceeded the MAX_TIME seconds, we stop. If we typically spend X seconds on an iteration, and we are less than X seconds away from MAX_TIME, we stop.
    while(elapsed < MAX_TIME && (elapsed + time_per_iteration) < MAX_TIME)
    {
        double iteration_start = omp_get_wtime();
 
        for(int i = 0; i < GRAPH_ORDER; i++)
        {
            new_pagerank[i] = 0.0;
        }
 
		for(int i = 0; i < GRAPH_ORDER; i++)
        {
			for(int j = 0; j < GRAPH_ORDER; j++)
            {
				if (adjacency_matrix[j][i] == 1.0)
                {
					int outdegree = 0;
				 
					for(int k = 0; k < GRAPH_ORDER; k++)
                    {
						if (adjacency_matrix[j][k] == 1.0)
                        {
							outdegree++;
						}
					}
					new_pagerank[i] += pagerank[j] / (double)outdegree;
				}
			}
		}
 
        for(int i = 0; i < GRAPH_ORDER; i++)
        {
            new_pagerank[i] = DAMPING_FACTOR * new_pagerank[i] + damping_value;
        }
 
        diff = 0.0;
        for(int i = 0; i < GRAPH_ORDER; i++)
        {
            diff += fabs(new_pagerank[i] - pagerank[i]);
        }
 
        for(int i = 0; i < GRAPH_ORDER; i++)
        {
            pagerank[i] = new_pagerank[i];
        }
 
		double iteration_end = omp_get_wtime();
		elapsed = omp_get_wtime() - start;
		iteration++;
		time_per_iteration = elapsed / iteration;
    }
    printf("%zu iterations achieved in %.2f seconds\n", iteration, elapsed);
}

int main(int argc, char* argv[])
{
    // We do not need argc, this line silences potential compilation warnings.
    (void) argc;
    // We do not need argv, this line silences potential compilation warnings.
    (void) argv;

    // Get the time at the very start.
    double start = omp_get_wtime();
    initialize_graph();
 
    // Initialise the (pseudo-)random number generator to a given seed, to guarantee reproducibility.
    srand(123);
    for(int i = 0; i < GRAPH_SIZE; i++)
    {
        int source = rand() % GRAPH_ORDER;
        int destination = rand() % GRAPH_ORDER;
        add_edge(source, destination);
    }
 
    /// The array in which each vertex pagerank is stored.
    double pagerank[GRAPH_ORDER];
    calculate_pagerank(pagerank);
 
    // Calculates the sum of all pageranks. It should be 1.0, so it can be used as a quick verification.
    double sum_ranks = 0.0;
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        if(i % 100 == 0)
        {
            printf("PageRank of Node %d: %.4lf\n", i, pagerank[i]);
        }
        sum_ranks += pagerank[i];
    }
    printf("Sum of all pageranks = %f.\n", sum_ranks);
    double end = omp_get_wtime();
 
    printf("Total time taken: %.2f seconds.\n", end - start);
 
    return 0;
}
