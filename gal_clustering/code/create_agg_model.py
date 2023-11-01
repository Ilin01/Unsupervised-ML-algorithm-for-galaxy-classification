import numpy as np
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier





def agg_model(run_number,threads,nr_hac_clusters,nodes,scaled_samples,samples_ids, samples_field):

    print('model '+str(run_number)+': AGG CLUSTERING HAS STARTED')

    rng = np.random.RandomState(0)

    classifier = RandomForestClassifier(n_jobs=threads,n_estimators=1000,random_state=rng,verbose=0)

    clusterer = AgglomerativeClustering(n_clusters=nr_hac_clusters)

    y=clusterer.fit_predict(nodes)
    classifier.fit(nodes,y)

 
    clust1_labels=np.empty(len(scaled_samples))
    slicee=1000
    values=np.array(range(0,len(scaled_samples),slicee))
    for sample in tqdm(values[:-1]):
        if sample==values[-2]:
            clust1_labels[sample:]=classifier.predict(scaled_samples[sample:])

        else:
            clust1_labels[sample:sample+slicee]=classifier.predict(scaled_samples[sample:sample+slicee])


    clust1_labels=np.array(clust1_labels).astype(int)


    print('model '+str(run_number)+': AGG CLUSTERING HAS FINISHED')

    return clust1_labels,clusterer.n_clusters_
