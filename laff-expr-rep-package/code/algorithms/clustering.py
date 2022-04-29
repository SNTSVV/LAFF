import pandas as pd
from math import sqrt
from kmodes.kmodes import KModes


class Clustering:

    @staticmethod
    def select_cluster_num(data, min_cls_num, max_cls_num):
        """
        Compute the optimal number of cluster of a given data set
        param data: the data set that we want to cluster
        param min_cls_num, max_cls_num: the possible range of the optimal number of cluster
        return n: optimal number of clusters (int)
        """
        # calculate the within clusters sum-of-squares (wcss) for different numbers of clusters
        wcss = []
        K = range(min_cls_num, max_cls_num)
        for k in K:
            kmode_model = KModes(n_clusters=k)
            kmode_model.fit(data)
            wcss.append(kmode_model.cost_)
        print('within clusters sum-of-squares: ' + str(wcss))
        n = Clustering.optimal_num_of_clusters(wcss, min_cls_num, max_cls_num)
        return n

    @staticmethod
    def optimal_num_of_clusters(wcss, min_cls_num, max_cls_num):
        x1, y1 = min_cls_num, wcss[0]
        x2, y2 = max_cls_num, wcss[len(wcss) - 1]

        distances = []
        for i in range(len(wcss)):
            x0 = i + min_cls_num
            y0 = wcss[i]
            numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
            denominator = sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            distances.append(numerator / denominator)

        return distances.index(max(distances)) + min_cls_num

    @staticmethod
    def cluster_on_fields(n, data_main, data_all):
        """
        cluster training set
        param n: number of clusters
        param data_main: data set containing only the main fields
        param data_all: data set containing all the fields
        return centroids: centroid of each cluster
        return cluster_list: values for each cluster
        return kmode_model: kmode object for prediction
        """
        n_clust= 0
        for j in range (n,1,-1):
            print("j  :" , j)
            kmode_model = KModes(n_clusters=j)
            clusters = kmode_model.fit_predict(data_main)
            # dataframe that contain the cluster number at the last column
            clusters = data_all.join(pd.Series(clusters, index=data_all.index, name='cluster'))

            centroids = pd.DataFrame(data=kmode_model.cluster_centroids_, columns=data_main.columns)
            print('number of rows in each cluster')
            print(clusters["cluster"].value_counts())

            # each element in the cluster_list contains the data regarding the same cluster
            clusters_list = []
            counter=0 #count the number of empty clusters
            for i in range(j):
                c_i = clusters.loc[clusters['cluster'] == i]
                c_i = c_i.drop(columns=['cluster'])
                clusters_list.append(c_i)
                print ("len c_i  :",len(c_i))
                if len(c_i)==0:
                    counter= counter + 1
            n_clust=j
            print("n of clusters:", n_clust)
            if counter == 0:
                break




        return centroids, clusters_list, kmode_model, n_clust
