#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PY_SSIZE_T_CLEAN

#include "Python.h"

typedef struct Node Node;
typedef struct Linked_list Linked_list;

/* - - - - - KMEANS ALGORITHM FUNCTION DEFINITIONS - - - - - */

int kmeans(double **d_vectors, double **centroids);

void update_single_centroid(double *centroid, Linked_list cluster, double **d_vectors);

void update_centroids(double **centroids, double **prev_centroids, Linked_list *clusters, double **d_vectors,
                      int *flag_stop);

void init_clusters(Linked_list **clusters);

void append_to_cluster(Linked_list *cluster, int d_vector_index);

void update_clusters(Linked_list **clusters, double **centroids, double **d_vectors);

void allocate_2D_array(void ***arr, int rows, int cols, size_t var_size);

void free_2D_array(void **arr, int rows);

void free_array_of_Linked_list(Linked_list *list, int rows);

void copy_K_first_d_vectors(double **copy_to, double **copy_from);

double d_distance(double *d_vector1, double *d_vector2);


/* - - - - - C PYTHON API FUNCTION DEFINITIONS - - - - - */
static PyObject *fit(PyObject *self, PyObject *args);

PyObject *build_PyCentroids(double **CCentroids);

PyObject *build_PyCentroids(double **CCentroids);

PyMODINIT_FUNC PyInit_mykmeanssp(void);

int parse_PyObject_to_2D_array(PyObject *PyArray_2D, double ***output_array_2D);


/* - - - - - GLOBAL VARIABLES - - - - - */
int K, iter, d, number_of_d_vectors;
double eps;

struct Node {
    int d_vector_index;
    Node *next;
};
struct Linked_list {
    Node *head;
    Node *tail;
    int size;
};

/* - - - - - C PYTHON API - - - - - */

static PyObject *fit(PyObject *self, PyObject *args) {
    PyObject *PyD_vectors, *PyCentroids;
    double **CD_vectors, **CCentroids;

    if (!PyArg_ParseTuple(args, "OOid", &PyD_vectors, &PyCentroids, &iter, &eps))
        return NULL;

    number_of_d_vectors = PyObject_Length(PyD_vectors);
    d = PyObject_Length(PyList_GetItem(PyD_vectors, 0));
    K = PyObject_Length(PyCentroids);
    if(parse_PyObject_to_2D_array(PyD_vectors, &CD_vectors) != 0)
        return NULL;

    if(parse_PyObject_to_2D_array(PyCentroids, &CCentroids) != 0)
        return NULL;

    /* Kmeans Algorithm */
    if (kmeans(CD_vectors, CCentroids) != 0)
        return NULL;

    /* CCentroids are the final centroids */
    PyCentroids = build_PyCentroids(CCentroids);
    free_2D_array((void **) CD_vectors, number_of_d_vectors);
    free_2D_array((void **) CCentroids, K);
    return PyCentroids;
}

PyObject *build_PyCentroids(double **CCentroids){
    PyObject *PyCentroids, *PyCentroid_single, *element;
    int r,c;
    PyCentroids = PyList_New(K);
    for(r = 0; r<K; r++){
        PyCentroid_single = PyList_New(d);
        for(c = 0; c<d; c++){
            element = PyFloat_FromDouble(CCentroids[r][c]);
            PyList_SET_ITEM(PyCentroid_single, c, element);
        }
        PyList_SET_ITEM(PyCentroids, r, PyCentroid_single);
    }
    return PyCentroids;
}

static PyMethodDef kmeansMethods[] = {
        {"fit",
            (PyCFunction) fit,
                 METH_VARARGS,
            PyDoc_STR("Kmeans Algorithm\nReturns the final centroids list.")},
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef kmeansmodule = {
        PyModuleDef_HEAD_INIT,
        "mykmeanssp",
        NULL,
        -1,
        kmeansMethods
};

PyMODINIT_FUNC PyInit_mykmeanssp(void) {
    return PyModule_Create(&kmeansmodule);
}

int parse_PyObject_to_2D_array(PyObject *PyArray_2D, double ***CArray_2D) {
    int rows, cols, r, c;
    PyObject *d_vector, *element;

    rows = PyObject_Length(PyArray_2D);
    if (rows == -1 || rows == 0)
        /* An error occurred while getting the length OR the array is empty */
        return 1;

    cols = PyObject_Length(PyList_GetItem(PyArray_2D, 0));
    /* Assuming the all rows carry the same length */
    if (cols == -1)
        return 1;

    /* Build CArray_2D from PyArray_2D */
    (*CArray_2D) = (double **) malloc(rows * sizeof(double *));
    if((*CArray_2D) == NULL)
        return 1;

    for (r = 0; r < rows; r++) {

        d_vector = PyList_GetItem(PyArray_2D, r);
        if (cols != PyObject_Length(d_vector))
        /*PyArray_2D must contain equal row's length for all rows*/
            return 1;


        (*CArray_2D)[r] = (double *) malloc(cols * sizeof(double));
        if((*CArray_2D)[r] == NULL)
            return 1;

        for (c = 0; c < cols; c++) {
            element = PyList_GetItem(d_vector, c);
            (*CArray_2D)[r][c] = PyFloat_AsDouble(element);
        }
    }
    return 0;
}

/* - - - - - KMEANS ALGORITHM - - - - - */

int kmeans(double **d_vectors, double **centroids) {
    int i, flag_stop;
    double **prev_centroids;
    Linked_list *clusters;

    /* Allocate dynamic memory for 2D arrays. */
    allocate_2D_array((void ***) &prev_centroids, K, d, sizeof(double));

    /* -- Kmeans Algorithm -- */
    flag_stop = 0;
    for (i = 0; i < iter && !flag_stop; i++) {
        if (i != 0)
            free_array_of_Linked_list(clusters, K);

        update_clusters(&clusters, centroids, d_vectors);
        update_centroids(centroids, prev_centroids, clusters, d_vectors, &flag_stop);
    }


    /* Free all dynamic memory created in the method. */
    free_array_of_Linked_list(clusters, K);
    free_2D_array((void **) prev_centroids, K);
    return 0;
}

/* - - - - - CENTROIDS - - - - - */

void update_single_centroid(double *centroid, Linked_list cluster, double **d_vectors) {
    int i, d_vector_index;
    double sum;
    Node *ptr_node;

    for (i = 0; i < d; i++) {
        sum = 0;
        ptr_node = cluster.head;
        while (ptr_node != NULL) {
            d_vector_index = ptr_node->d_vector_index;
            sum += d_vectors[d_vector_index][i];
            ptr_node = ptr_node->next;
        }
        centroid[i] = sum / cluster.size;
    }
}

/*
  Calculates the new values of the centroids.
  flag_stop=1 if all centroids moved a distance less than eps (epsilon). o.w flag_stop=0
 */
void update_centroids(double **centroids, double **prev_centroids, Linked_list *clusters, double **d_vectors,
                      int *flag_stop) {

    int c;
    *flag_stop = 1; /* flag_stop is 1 iff all centroids changed to distance smaller than eps. */

    copy_K_first_d_vectors(prev_centroids, centroids);
    for (c = 0; c < K; c++) {
        update_single_centroid(centroids[c], clusters[c], d_vectors);

        if (*flag_stop == 0 || d_distance(prev_centroids[c], centroids[c]) >= eps)
            *flag_stop = 0;
    }
}

/* - - - - - CLUSTERS - - - - - */

void init_clusters(Linked_list **clusters) {
    int k;
    *clusters = (Linked_list *) malloc(K * sizeof(Linked_list));
    for (k = 0; k < K; k++) {
        /* Empty linked list */
        (*clusters)[k].head = NULL;
        (*clusters)[k].tail = NULL;
        (*clusters)[k].size = 0;
    }
}

void append_to_cluster(Linked_list *cluster, int d_vector_index) {
    /* Set new node object in memory */
    Node *new_node = (Node *) malloc(sizeof(Node));
    new_node->d_vector_index = d_vector_index;
    new_node->next = NULL;

    /* Append new node to linked list */

    cluster->size++;
    if (cluster->head == NULL) {
        cluster->head = new_node;
        cluster->tail = new_node;

    } else {
        cluster->tail->next = new_node;
        cluster->tail = cluster->tail->next;
    }
}

void update_clusters(Linked_list **clusters, double **centroids, double **d_vectors) {
    int v, k, min_cluster_index;
    double min_d_distance, curr_d_distance;
    /* Old information of the clusters is not important, initialize them as new. */
    init_clusters(clusters);

    for (v = 0; v < number_of_d_vectors; v++) {

        min_d_distance = -1;
        min_cluster_index = 0;
        for (k = 0; k < K; k++) {
            curr_d_distance = d_distance(d_vectors[v], centroids[k]);
            if (min_d_distance == -1 || curr_d_distance < min_d_distance) {
                min_d_distance = curr_d_distance;
                min_cluster_index = k;
            }
        }
        /* Add d_vectors[v] to min_cluster_index cluster. */
        append_to_cluster(&((*clusters)[min_cluster_index]), v);
    }
}

/* - - - - - DYNAMIC MEMORY HANDLE - - - - - */

void allocate_2D_array(void ***arr, int rows, int cols, size_t var_size) {
    int i;
    *arr = (void **) malloc(rows * sizeof(void *));
    for (i = 0; i < rows; i++)
        (*arr)[i] = (void *) malloc(cols * sizeof(var_size));
}

void free_2D_array(void **arr, int rows) {
    int i;
    for (i = 0; i < rows; i++)
        free(arr[i]);
    free(arr);
}

void free_array_of_Linked_list(Linked_list *list, int rows) {
    int i;
    Node *next, *curr;
    for (i = 0; i < rows; i++) {
        curr = list[i].head;
        while (curr != NULL) {
            next = curr->next;
            free(curr);
            curr = next;
        }
    }
    free(list);
}

/* - - - - - FUNCTIONS - - - - - */

void copy_K_first_d_vectors(double **copy_to, double **copy_from) {
    /* copy_from could be d_vectors or a different copy_to 2D array */
    int i, j;
    for (i = 0; i < K; i++) {
        /* Assign K first arrays of copy_from as copy_to */
        for (j = 0; j < d; j++)
            copy_to[i][j] = copy_from[i][j];
    }
}

double d_distance(double *d_vector1, double *d_vector2) {
    int i;
    double d_distance;
    d_distance = 0;
    for (i = 0; i < d; i++)
        d_distance += pow(d_vector2[i] - d_vector1[i], 2);
    return sqrt(d_distance);
}