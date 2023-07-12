/**
 * @file main.f08
 * @brief This file provides you with the original implementation of pagerank.
 * Your challenge is to optimise it using OpenMP and/or MPI.
 * @author Ludovic Capelli (l.capelli@epcc.ed.ac.uk)
 **/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

/// The number of vertices in the graph.
#define GRAPH_ORDER 1000
/// Parameters used in pagerank convergence, do not change.
#define DAMPING_FACTOR 0.85
/// The number of seconds to not exceed forthe calculation loop.
#define MAX_TIME 10

int MPI_RANK, MPI_SIZE;

/**
 * @brief Indicates which vertices are connected.
 * @details If an edge links vertex A to vertex B, then adjacency_matrix[A][B]
 * will be 1.0. The absence of edge is represented with value 0.0.
 * Redundant edges are still represented with value 1.0.
 */
double adjacency_matrix[GRAPH_ORDER][GRAPH_ORDER];
double max_diff = 0.0;
double min_diff = 1.0;
double total_diff = 0.0;

int offsets[GRAPH_ORDER+1];
int indices[GRAPH_ORDER*GRAPH_ORDER];

int CSR_ROW_OFFSETS[3];

/* 0, 293 */

/* convert global row to local row index for the CSR row. */
int csr_row_g2i(int row) {
  return row - CSR_ROW_OFFSETS[MPI_RANK];
}

void initialize_graph(void)
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

 * The pagerank and new_pagerank arrays are distributed as follows:
 * [0, ... , CSR_ROW_OFFSETS[0], ..., CSR_ROW_OFFSETS[1]]
 *      ^ rank0                   ^ rank1
 */
void calculate_pagerank(double pagerank[])
{
  double initial_rank = 1.0 / GRAPH_ORDER;
  double new_pagerank[GRAPH_ORDER], local_new_pagerank[GRAPH_ORDER];

  double local_pagerank_total = 0, global_pagerank_total;
  for(int i = 0; i < GRAPH_ORDER; ++i) {
    // Initialise all vertices to 1/n.
    pagerank[i] = initial_rank;
    new_pagerank[i] = 0.0;
  }

  double damping_value = (1.0 - DAMPING_FACTOR) / GRAPH_ORDER;
  double local_diff, global_diff;
  size_t iteration = 0;
  double start = omp_get_wtime();
  double global_elapsed = omp_get_wtime() - start;
  double time_per_iteration = 0;

  // If we exceeded the MAX_TIME seconds, we stop. If we typically spend X seconds on an iteration, and we are less than X seconds away from MAX_TIME, we stop.
  while(global_elapsed < MAX_TIME && (global_elapsed + time_per_iteration) < MAX_TIME) {
    double iteration_start = omp_get_wtime();

    for(int i = 0; i < GRAPH_ORDER; i++) {
      new_pagerank[i] = 0.0;
      local_new_pagerank[i] = 0.0;
    }

    for (int j = CSR_ROW_OFFSETS[MPI_RANK]; j < CSR_ROW_OFFSETS[MPI_RANK+1]; ++j) {
      int local_j = csr_row_g2i(j);

      for (int i = offsets[local_j]; i < offsets[local_j+1]; ++i) {
        int i_node = indices[i];
        int outdegree = offsets[local_j+1] - offsets[local_j];
        local_new_pagerank[i_node] += pagerank[j] / (double)outdegree;
      }
    }

    MPI_Allreduce(local_new_pagerank, new_pagerank, GRAPH_ORDER,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    local_diff = 0.0;
    double local_pagerank_total = 0.0, global_pagerank_total;

    for (int i = 0; i < GRAPH_ORDER; ++i) {
      new_pagerank[i] = DAMPING_FACTOR * new_pagerank[i] + damping_value;
      local_diff += fabs(new_pagerank[i] - pagerank[i]);
      pagerank[i] = new_pagerank[i];
      local_pagerank_total += pagerank[i];
    }
    global_diff = local_diff;
    global_pagerank_total = local_pagerank_total;

    max_diff = (max_diff < global_diff) ? global_diff : max_diff;
    total_diff += global_diff;
    min_diff = (min_diff > global_diff) ? global_diff : min_diff;

    if(fabs(global_pagerank_total - 1.0) >= 1.0) {
      printf("[ERROR] Iteration %zu: sum of all pageranks is not 1 but %.12f.\n",
             iteration, global_pagerank_total);
    }

    double iteration_end = omp_get_wtime();
    double local_elapsed = omp_get_wtime() - start;
    MPI_Allreduce(&local_elapsed, &global_elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    iteration++;
    time_per_iteration = global_elapsed / iteration;
  }

  printf("%zu iterations achieved in %.2f seconds MPI_RANK: %d\n",
         iteration, global_elapsed, MPI_RANK);
}

/**
 * @brief Populates the edges in the graph for testing.
 **/
void generate_nice_graph(void)
{
  printf("Generate a graph for testing purposes (i.e.: a nice and conveniently designed graph :) )\n");
  double start = omp_get_wtime();
  initialize_graph();
  for(int i = 0; i < GRAPH_ORDER; i++)
    {
      for(int j = 0; j < GRAPH_ORDER; j++)
        {
          int source = i;
          int destination = j;
          if(i != j)
            {
              adjacency_matrix[source][destination] = 1.0;
            }
        }
    }
  printf("%.2f seconds to generate the graph.\n", omp_get_wtime() - start);
}

/**
 * @brief Populates the edges in the graph for the challenge.
 **/
void generate_sneaky_graph(void)
{
  printf("Generate a graph for the challenge (i.e.: a sneaky graph :P )\n");
  double start = omp_get_wtime();
  initialize_graph();
  int csr_index = 0;
  offsets[0] = 0;
  for(int i = CSR_ROW_OFFSETS[MPI_RANK]; i < CSR_ROW_OFFSETS[MPI_RANK+1]; i++) {
    int non_zeros = 0;
    for(int j = 0; j < GRAPH_ORDER - i; j++) {
      int destination = j;
      if(i != j) {
        indices[csr_index] = destination;
        non_zeros++;
        csr_index++;
      }
    }

    int local_i = csr_row_g2i(i);
    offsets[local_i+1] = offsets[local_i] + non_zeros;
  }
  printf("%.2f seconds to generate the graph.\n", omp_get_wtime() - start);

}

int main(int argc, char* argv[])
{
  // We do not need argc, this line silences potential compilation warnings.
  (void) argc;
  // We do not need argv, this line silences potential compilation warnings.
  (void) argv;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &MPI_SIZE);
  MPI_Comm_rank(MPI_COMM_WORLD, &MPI_RANK);

  CSR_ROW_OFFSETS[0] = 0;

  if (MPI_SIZE == 1) {
    CSR_ROW_OFFSETS[1] = 1000;
  }
  else if (MPI_SIZE == 2) {
    CSR_ROW_OFFSETS[1] = 293;
    CSR_ROW_OFFSETS[2] = 1000;
  }

  printf("This program has two graph generators: generate_nice_graph and generate_sneaky_graph. If you intend to submit, your code will be timed on the sneaky graph, remember to try both.\n");

  // Get the time at the very start.
  double start = omp_get_wtime();

  generate_sneaky_graph();

  /// The array in which each vertex pagerank is stored.
  double pagerank[GRAPH_ORDER];
  calculate_pagerank(pagerank);

  // Calculates the sum of all pageranks. It should be 1.0, so it can be used as a quick verification.
  double local_sum_ranks = 0.0, global_sum_ranks;
  for(int i = CSR_ROW_OFFSETS[MPI_RANK]; i < CSR_ROW_OFFSETS[MPI_RANK+1]; i++) {
    if(i % 100 == 0) {
        printf("PageRank of vertex %d: %.6f\n", i, pagerank[i]);
    }
    local_sum_ranks += pagerank[i];
  }
  MPI_Allreduce(&local_sum_ranks, &global_sum_ranks, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (MPI_RANK == 0) {
    printf("Sum of all pageranks = %.12f, total diff = %.12f, max diff = %.12f and min diff = %.12f.\n", global_sum_ranks, total_diff, max_diff, min_diff);
  }
  double end = omp_get_wtime();

  printf("Total time taken: %.2f seconds.\n", end - start);

  MPI_Finalize();

  return 0;
}
