#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include <limits.h>
#define LINE 1024 * 1024 * 100

typedef struct Node
{
    double value;
    unsigned long index;
} Node;

unsigned long get_columns(char *string)
{
    unsigned long columns = 0;
    while (*string)
    {
        if (*string == '\t')
        {
            columns++;
        }
        string++;
    }
    return columns;
}

int read_file(char *file, char ***row_names, double ***data, unsigned long *rows, unsigned long *columns)
{
    char *buffer = malloc(sizeof(char) * LINE);
    fpos_t file_body;
    FILE *open_file = fopen(file, "r");
    fgets(buffer, LINE, open_file); // Header.
    assert(buffer[strlen(buffer) - 1] == '\n');
    fgetpos(open_file, &file_body);
    *columns = get_columns(buffer); // #columns

    while (fgets(buffer, LINE, open_file))
    {
        unsigned long buffer_size = strlen(buffer);
        assert(buffer_size < LINE - 1 || buffer[buffer_size - 1] == '\n');
        (*rows)++;
    }

    *row_names = malloc(sizeof(char *) * (*rows));
    *data = malloc(sizeof(double *) * (*rows));
    for (unsigned long row_index = 0; row_index < *rows; row_index++)
    {
        (*data)[row_index] = malloc(sizeof(double) * (*columns));
    }

    unsigned long row_index = 0;
    fsetpos(open_file, &file_body);
    while (fgets(buffer, LINE, open_file))
    {
        char *sep = buffer;
        unsigned long column_index = 0;
        char *element = strsep(&sep, "\t"); // Read row name.
        (*row_names)[row_index] = malloc(sizeof(char) * (strlen(element) + 1));
        strcpy((*row_names)[row_index], element);
        while ((element = strsep(&sep, "\t")))
        {
            (*data)[row_index][column_index] = atof(element);
            column_index++;
        }
        row_index++;
    }
    fclose(open_file);
    free(buffer);
    return 0;
}

double calculate_pairwise_distance(double *x1, double *x2, unsigned long size)
{
    double distance = 0;
    for (unsigned long index = 0; index < size; index++)
    {
        distance += pow(x1[index] - x2[index], 2.0);
    }
    distance = pow(distance, 0.5);
    return distance;
}

double **calculate_distance_matrix(double **data, unsigned long samples, unsigned long features)
{
    /* Samples by samples L-matrix. */
    double **distance_matrix = malloc(sizeof(double *) * samples);

    unsigned long samples_ = 0;
    for (unsigned long sample_index = 0; sample_index < samples; sample_index++)
    {
        distance_matrix[sample_index] = malloc(sizeof(double) * (++samples_));
    }

    for (unsigned long sample_row_index = 0; sample_row_index < samples; sample_row_index++)
    {
        for (unsigned long sample_column_index = 0; sample_column_index <= sample_row_index; sample_column_index++)
        {
            if (sample_row_index == sample_column_index)
            {
                distance_matrix[sample_row_index][sample_column_index] = 0.0;
            }
            else
            {
                distance_matrix[sample_row_index][sample_column_index] = calculate_pairwise_distance(data[sample_row_index], data[sample_column_index], features);
            }
        }
    }

    return distance_matrix;
}

int free_data(double **data, unsigned long rows)
{
    for (unsigned long row_index = 0; row_index < rows; row_index++)
    {
        free(data[row_index]);
    }
    free(data);
    return 0;
}

double get_distance(double **distance_matrix, unsigned long x1, unsigned long x2)
{
    if (x1 <= x2)
    {
        return distance_matrix[x2][x1];
    }
    else
    {
        return distance_matrix[x1][x2];
    }
}

int compare(const void *x1, const void *x2)
{
    return ((Node *)x1)->value <= ((Node *)x2)->value ? -1 : 1;
}

int initialize_medoids(double **distance_matrix, unsigned long samples, unsigned long clusters, unsigned long *medoids, Node *v)
{
    /* Ref: A simple and fast algorithm for K-medoids clustering. */
    for (unsigned long j = 0; j < samples; j++)
    {
        (v + j)->value = 0.0; // Add v[j]
        (v + j)->index = j;
        for (unsigned long i = 0; i < samples; i++)
        {
            double dij = get_distance(distance_matrix, i, j);
            double di = 0.0;
            for (unsigned long l = 0; l < samples; l++)
            {
                di += get_distance(distance_matrix, i, l);
            }
            (v + j)->value += dij / (di + 1e-10);
        }
    }
    qsort(v, samples, sizeof(Node), compare);
    for (unsigned long cluster_index = 0; cluster_index < clusters; cluster_index++)
    {
        medoids[cluster_index] = (v + cluster_index)->index;
    }
    return 0;
}

int assign_label(double **distance_matrix, unsigned long *medoids, unsigned long *labels, unsigned long samples, unsigned long clusters)
{
    for (unsigned long sample_index = 0; sample_index < samples; sample_index++)
    {
        double min_distance = __DBL_MAX__;
        for (unsigned long cluster_index = 0; cluster_index < clusters; cluster_index++)
        {
            double distance = get_distance(distance_matrix, sample_index, medoids[cluster_index]);
            // printf("sample1: %lu, sample2: %lu, distance: %lf, min_distance: %lf\n", sample_index, medoids[cluster_index], distance, min_distance);
            if (distance < min_distance)
            {
                labels[sample_index] = cluster_index;
                min_distance = distance;
            }
        }
    }
    return 0;
}

double calculate_total_distance(double **distance_matrix, unsigned long *medoids, unsigned long *labels, unsigned long samples)
{
    double total_distance = 0.0;
    for (unsigned long sample_index = 0; sample_index < samples; sample_index++)
    {
        total_distance += get_distance(distance_matrix, sample_index, medoids[labels[sample_index]]);
    }
    return total_distance;
}

int update_medoids(double **distance_matrix, unsigned long *medoids, unsigned long samples, unsigned long *labels, unsigned long clusters, Node *nodes)
{
    for (unsigned long cluster_index = 0; cluster_index < clusters; cluster_index++)
    {
        unsigned long cluster_samples = 0;
        for (unsigned long sample_index = 0; sample_index < samples; sample_index++)
        {
            if (labels[sample_index] == cluster_index)
            {
                (nodes + cluster_samples)->value = 0.0;
                (nodes + cluster_samples)->index = sample_index;
                cluster_samples++;
            }
        }
        for (unsigned long sample_index1 = 0; sample_index1 < cluster_samples; sample_index1++)
        {
            for (unsigned long sample_index2 = 0; sample_index2 < cluster_samples; sample_index2++)
            {
                (nodes + sample_index1)->value += get_distance(distance_matrix, (nodes + sample_index1)->index, (nodes + sample_index2)->index);
            }
        }
        double min_distance = __DBL_MAX__;
        for (unsigned long sample_index = 0; sample_index < cluster_samples; sample_index++)
        {
            if ((nodes + sample_index)->value < min_distance)
            {
                medoids[cluster_index] = (nodes + sample_index)->index;
                min_distance = (nodes + sample_index)->value;
            }
        }
    }
    return 0;
}

int main(int argc, char *argv[])
{
    char file[10240];
    unsigned long clusters = 2;

    int necessary_parameters = 0;
    for (int index = 1; index < argc; index++)
    {
        if (!strncmp(argv[index], "-i", 2))
        {
            assert(index + 1 < argc);
            strncpy(file, argv[index + 1], 10240);
            file[10239] = 0;
            necessary_parameters++;
        }
        else if (!strncmp(argv[index], "-c", 2) || !strncmp(argv[index], "-k", 2))
        {
            assert(index + 1 < argc);
            sscanf(argv[index + 1], "%lu", &clusters);
            necessary_parameters++;
        }
    }

    if (necessary_parameters != 2)
    {
        printf("Usage:\n%s -input FILE -clusters INT\n", argv[0]);
        exit(0);
    }

    char **sample_names = NULL;
    double **data = NULL;
    unsigned long samples = 0;
    unsigned long features = 0;

    read_file(file, &sample_names, &data, &samples, &features);

    double **distance_matrix = calculate_distance_matrix(data, samples, features);
    free_data(data, samples);
    assert(samples >= clusters);

    Node *nodes = malloc(sizeof(Node) * samples); // Not initialization.

    unsigned long *medoids = malloc(sizeof(unsigned long) * clusters);
    initialize_medoids(distance_matrix, samples, clusters, medoids, nodes);

    unsigned long *labels = malloc(sizeof(unsigned long) * samples);

    double total_distance_ = 0.0;
    while (true)
    {
        assign_label(distance_matrix, medoids, labels, samples, clusters);
        double total_distance = calculate_total_distance(distance_matrix, medoids, labels, samples);
        if (total_distance != total_distance_)
        {
            update_medoids(distance_matrix, medoids, samples, labels, clusters, nodes);
            total_distance_ = total_distance;
        }
        else
        {
            break;
        }
    }

    printf("Sample\tCluster\n");
    for (unsigned long sample_index = 0; sample_index < samples; sample_index++)
    {
        printf("%s\t%lu\n", sample_names[sample_index], labels[sample_index]);
    }

    free(nodes);
    free(medoids);
    free(sample_names);
    free(labels);
    return 0;
}