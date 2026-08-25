#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (argc != 6)
    {
        if (rank == 0)
        {
            printf("Usage: ./src M D1 D2 T seed\n");
        }
        MPI_Finalize();
        return 0;
    }

    int M = atoi(argv[1]);
    int D1 = atoi(argv[2]);
    int D2 = atoi(argv[3]);
    int T = atoi(argv[4]);
    int seed = atoi(argv[5]);

    double *buffer_D1 = (double *)malloc(M * sizeof(double));
    double *buffer_D2 = (double *)malloc(M * sizeof(double));
    double *recv_D1 = (double *)malloc(M * sizeof(double));
    double *recv_D2 = (double *)malloc(M * sizeof(double));
    double *temp = (double *)malloc(M * sizeof(double));

    srand(seed);
    for (int i = 0; i < M; i++)
    {
        double val = (double)rand() * (rank + 1) / 10000.0;
        buffer_D1[i] = val;
        buffer_D2[i] = val;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (int t = 0; t < T; t++)
    {
        int valid_sender = (rank + D1 < P);
        int valid_D1 = valid_sender;
        int valid_D2 = valid_sender && (rank + D2 < P);

        // D1 even-odd communication
        if (rank % 2 == 0)
        {
            if (valid_D1)
            {
                MPI_Send(buffer_D1, M, MPI_DOUBLE, rank + D1, 0, MPI_COMM_WORLD);
            }

            if (rank - D1 >= 0)
            {
                MPI_Recv(temp, M, MPI_DOUBLE, rank - D1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                for (int i = 0; i < M; i++)
                {
                    temp[i] = temp[i] * temp[i];
                }

                MPI_Send(temp, M, MPI_DOUBLE, rank - D1, 2, MPI_COMM_WORLD);
            }

            if (valid_D1)
                MPI_Recv(recv_D1, M, MPI_DOUBLE, rank + D1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else
        {
            if (rank - D1 >= 0)
            {
                MPI_Recv(temp, M, MPI_DOUBLE, rank - D1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                for (int i = 0; i < M; i++)
                {
                    temp[i] = temp[i] * temp[i];
                }

                MPI_Send(temp, M, MPI_DOUBLE, rank - D1, 2, MPI_COMM_WORLD);
            }

            if (valid_D1)
            {
                MPI_Send(buffer_D1, M, MPI_DOUBLE, rank + D1, 0, MPI_COMM_WORLD);
            }

            if (valid_D1)
            {
                MPI_Recv(recv_D1, M, MPI_DOUBLE, rank + D1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        // D2 even-odd communication
        if (rank % 2 == 0)
        {
            if (valid_D2)
            {
                MPI_Send(buffer_D2, M, MPI_DOUBLE, rank + D2, 1, MPI_COMM_WORLD);
            }

            if (rank - D2 >= 0)
            {
                MPI_Recv(temp, M, MPI_DOUBLE, rank - D2, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                for (int i = 0; i < M; i++)
                {
                    temp[i] = log(temp[i]);
                }

                MPI_Send(temp, M, MPI_DOUBLE, rank - D2, 3, MPI_COMM_WORLD);
            }

            if (valid_D2)
            {
                MPI_Recv(recv_D2, M, MPI_DOUBLE, rank + D2, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        else
        {
            if (rank - D2 >= 0)
            {
                MPI_Recv(temp, M, MPI_DOUBLE, rank - D2, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                for (int i = 0; i < M; i++)
                {
                    temp[i] = log(temp[i]);
                }

                MPI_Send(temp, M, MPI_DOUBLE, rank - D2, 3, MPI_COMM_WORLD);
            }

            if (valid_D2)
            {
                MPI_Send(buffer_D2, M, MPI_DOUBLE, rank + D2, 1, MPI_COMM_WORLD);
            }

            if (valid_D2)
            {
                MPI_Recv(recv_D2, M, MPI_DOUBLE, rank + D2, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        // Data update
        if (valid_D1)
        {
            for (int i = 0; i < M; i++)
            {
                buffer_D1[i] = (unsigned long long)(recv_D1[i]) % 100000;
            }
        }

        if (valid_D2)
        {
            for (int i = 0; i < M; i++)
            {
                buffer_D2[i] = recv_D2[i] * 100000;
            }
        }
    }

    double maxD1 = -1e18, maxD2 = -1e18;
    int valid_sender = (rank + D1 < P);

    if (rank + D1 < P)
    {
        for (int i = 0; i < M; i++)
        {
            if (recv_D1[i] > maxD1)
            {
                maxD1 = recv_D1[i];
            }
        }
    }

    if (rank + D2 < P)
    {
        for (int i = 0; i < M; i++)
        {
            if (recv_D2[i] > maxD2)
            {
                maxD2 = recv_D2[i];
            }
        }
    }

    if (rank != 0 && valid_sender)
    {
        MPI_Send(&maxD1, 1, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD);
        MPI_Send(&maxD2, 1, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD);
    }

    double globalMaxD1 = -1e18, globalMaxD2 = -1e18;

    // Aggregate results at rank 0
    if (rank == 0)
    {
        if (valid_sender)
        {
            if (maxD1 > globalMaxD1)
            {
                globalMaxD1 = maxD1;
            }
            if (maxD2 > globalMaxD2)
            {
                globalMaxD2 = maxD2;
            }
        }

        for (int r = 1; r < P; r++)
        {
            if (r + D1 < P)
            {
                double t1, t2;
                MPI_Recv(&t1, 1, MPI_DOUBLE, r, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&t2, 1, MPI_DOUBLE, r, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (t1 > globalMaxD1)
                {
                    globalMaxD1 = t1;
                }
                if (t2 > globalMaxD2)
                {
                    globalMaxD2 = t2;
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    double localTime = end - start;
    double totalTime;

    MPI_Reduce(&localTime, &totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("%lf %lf %lf\n", globalMaxD1, globalMaxD2, totalTime);
    }

    free(buffer_D1);
    free(buffer_D2);
    free(recv_D1);
    free(recv_D2);
    free(temp);

    MPI_Finalize();
    return 0;
}
