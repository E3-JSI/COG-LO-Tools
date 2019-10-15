import numpy as np

def discretesample(p,n):
    return np.random.choice(len(p), n, p=p)

def kMeansSphere(X, k):
    n = X.shape[0]
    d = X.shape[1]

    def similarity(x1, x2):
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        return np.dot(x1, x2) / (norm_x1 * norm_x2)

    def calc_centroid(X):
        return np.mean(X, 0)

    def count_diff(a, prev_a):
        diff = np.abs(a - prev_a)
        diff[diff > 1] = 1

        return np.sum(diff)

    samples = np.random.randint(0, high=n, size=k)
    C = X[samples, :]
    C = np.zeros((k, d))  # ???
    rands = np.random.randint(0, n)

    C[0, :] = X[rands]

    probs = np.zeros(n)

    for centroidN in range(1, k):
        for recN in range(0, n):
            rec = X[recN, :]
            nearest_dist = np.inf

            for exist_centroidN in range(0, centroidN):
                centroid = C[exist_centroidN, :]
                clust_rec_sim = similarity(rec, centroid)
                clust_rec_dist = 0.5 * (1 - clust_rec_sim)

                if clust_rec_dist < nearest_dist:
                    nearest_dist = clust_rec_dist
            probs[recN] = nearest_dist * nearest_dist
        norm_factor = 1.0 / np.sum(probs)
        probs = probs * norm_factor

        chosenN = discretesample(probs, 1)[0]
        C[centroidN, :] = X[chosenN, :]

        print('Chosen centroid {}, probability: {}, max probability: {} '.format(chosenN, probs[centroidN],
                                                                                 probs.max()))

    prev_assignment = np.zeros(n)
    assignment = np.zeros(n)
    assignment.fill(0)
    prev_assignment.fill(0)

    change = True

    while change:
        for recN in range(0, n):
            xi = X[recN, :]

            best_idx = -1
            best_sim = np.NINF

            for clustN in range(0, k):
                sim = similarity(xi, C[clustN, :])

                if sim > best_sim:
                    best_idx = clustN
                    best_sim = sim

            assignment[recN] = best_idx

        for clustN in range(0, k):
            Yn = X[assignment == clustN, :]
            C[clustN, :] = calc_centroid(Yn)

        diff = count_diff(assignment, prev_assignment)
        change = diff > 0
        tmp = prev_assignment
        prev_assignment = assignment
        assignment = tmp
        print('Diff', diff)


    print("kMeans done")
    return assignment
