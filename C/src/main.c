/**
 * @file main.f08
 * @brief This file provides you with the original implementation of pagerank.
 * Your challenge is to optimise it using OpenMP and/or MPI.
 * @author Ludovic Capelli (l.capelli@epcc.ed.ac.uk)
 **/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
/* #include <mpi.h> */

/// The number of vertices in the graph.
#define GRAPH_ORDER 1000
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
float max_diff = 0.0;
float min_diff = 1.0;
float total_diff = 0.0;

void initialize_graph(int* offsets, int* indices)
{
  for(int i = 0; i < GRAPH_ORDER; i++)
    {
      offsets[i] = 0;
      for(int j = 0; j < GRAPH_ORDER; j++)
        {
          /* adjacency_matrix[i][j] = 0.0; */
          indices[i * GRAPH_ORDER + j] = 0;
        }
    }
}

/**
 * @brief Calculates the pagerank of all vertices in the graph.
 * @param pagerank The array in which store the final pageranks.
 */
inline void calculate_pagerank(int* offsets, int* indices, float *pagerank)
{
  float initial_rank;
  #pragma omp parallel
  {
    initial_rank = 1.0 / GRAPH_ORDER;
  }
  float new_pagerank[GRAPH_ORDER];
  for(int i = 0; i < GRAPH_ORDER; i++) {
    // Initialise all vertices to 1/n.
    pagerank[i] = initial_rank;
    new_pagerank[i] = 0.0;
  }

  float damping_value = (1.0 - DAMPING_FACTOR) / GRAPH_ORDER;
  float diff = 1.0;
  size_t iteration = 0;
  double start = omp_get_wtime();
  double elapsed = omp_get_wtime() - start;
  double time_per_iteration = 0;


  // If we exceeded the MAX_TIME seconds, we stop. If we typically spend X seconds on an iteration, and we are less than X seconds away from MAX_TIME, we stop.

  while(elapsed < MAX_TIME && (elapsed + time_per_iteration) < MAX_TIME) {

    double iteration_start = omp_get_wtime();

    memset(new_pagerank, 0, sizeof(float) * GRAPH_ORDER);

#pragma omp parallel for reduction(+:new_pagerank[0:GRAPH_ORDER])
    for (int j = 0; j < GRAPH_ORDER; ++j) {
      int col_end = offsets[j+1];
      int col_start = offsets[j];
      float outdegree =  1.0 / (col_end - col_start);
      float pagerank_j = pagerank[j];
      for (int i = col_start; i < col_end; ++i) {
        int i_node = indices[i];
        new_pagerank[i_node] += pagerank_j * outdegree;
      }
    }

    diff = 0.0;
    float pagerank_total = 0.0;

#pragma omp parallel for
    for(int i = 0; i < GRAPH_ORDER; i++) {
      new_pagerank[i] = DAMPING_FACTOR * new_pagerank[i] + damping_value;
    }

#pragma omp parallel for reduction(+:pagerank_total) reduction(+:diff)
    for (int i = 0; i < GRAPH_ORDER; ++i) {
      diff += fabs(new_pagerank[i] - pagerank[i]);
      pagerank_total += new_pagerank[i];
    }

    memcpy(pagerank, new_pagerank, sizeof(float) * GRAPH_ORDER);

    max_diff = (max_diff < diff) ? diff : max_diff;
    total_diff += diff;
    min_diff = (min_diff > diff) ? diff : min_diff;


    if(fabs(pagerank_total - 1.0) >= 1.0) {
      printf("[ERROR] Iteration %zu: sum of all pageranks is not 1 but %.12f.\n",
             iteration, pagerank_total);
    }

    double iteration_end = omp_get_wtime();
    elapsed = omp_get_wtime() - start;
    iteration++;
    time_per_iteration = elapsed / iteration;
  }

  printf("%zu iterations achieved in %.2f seconds\n", iteration, elapsed);
}


/**
 * @brief Populates the edges in the graph for the challenge.
 **/
inline void generate_sneaky_graph(int *offsets, int *indices)
{
  /* printf("Generate a graph for the challenge (i.e.: a sneaky graph :P )\n"); */
  double start = omp_get_wtime();
  initialize_graph(offsets, indices);
  int csr_index = 0;
  offsets[0] = 0;
  for(int i = 0; i < GRAPH_ORDER; i++) {
    int non_zeros = 0;
    for(int j = 0; j < GRAPH_ORDER - i; ++j) {
      int destination = j;
      if(i != j) {
        indices[csr_index] = destination;
        non_zeros++;
        csr_index++;
      }
    }
    offsets[i+1] = offsets[i] + non_zeros;
  }
  /* printf("%.2f seconds to generate the graph.\n", omp_get_wtime() - start); */
}

int main(int argc, char* argv[])
{
  // We do not need argc, this line silences potential compilation warnings.
  (void) argc;
  // We do not need argv, this line silences potential compilation warnings.
  (void) argv;

  printf("This program has two graph generators: generate_nice_graph and generate_sneaky_graph. If you intend to submit, your code will be timed on the sneaky graph, remember to try both.\n");
  double sum_ranks;

  int *offsets, *indices;
  offsets = (int*)malloc(sizeof(int) * (GRAPH_ORDER+1));
  indices = (int*)malloc(sizeof(int) * (GRAPH_ORDER * GRAPH_ORDER));

  // Get the time at the very start.
  double start = omp_get_wtime();

  /* #pragma omp target data map(to: offsets[0:GRAPH_ORDER+1]) map(to: indices[GRAPH_ORDER*GRAPH_ORDER]) map(tofrom: sum_ranks) map(tofrom: max_diff) map(tofrom: min_diff) map(tofrom: total_diff) */
  /*   { */

  generate_sneaky_graph(offsets, indices);

  /// The array in which each vertex pagerank is stored.
  float pagerank[GRAPH_ORDER];
  calculate_pagerank(offsets, indices, pagerank);

  // Calculates the sum of all pageranks. It should be 1.0, so it can be used as a quick verification.
  sum_ranks = 0.0;
  for(int i = 0; i < GRAPH_ORDER; i++) {
    if(i % 100 == 0) {
      printf("PageRank of vertex %d: %.6f\n", i, pagerank[i]);
    }
    sum_ranks += pagerank[i];
  }
  /* } */

  printf("Sum of all pageranks = %.12f, total diff = %.12f, max diff = %.12f and min diff = %.12f.\n", sum_ranks, total_diff, max_diff, min_diff);

  double end = omp_get_wtime();

  printf("Total time taken: %.2f seconds.\n", end - start);

  return 0;
}
