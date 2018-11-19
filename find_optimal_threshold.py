from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import sys

l = open(sys.argv[1]).read()[:-1].split('\n')

distances = [] # squared L2 distance between pairs
identical = [] # 1 if same identity, 0 otherwise

for line in l:
    (person1, person2, dist) = line.split('\t')
    distances.append(float(dist))
    identical.append(1 if person1.split('/')[0] == person2.split('/')[0] else 0)

distances = np.array(distances)
identical = np.array(identical)

thresholds = np.arange(0.3, 2, 0.01)

f1_scores = [f1_score(identical, distances < t) for t in thresholds]
acc_scores = [accuracy_score(identical, distances < t) for t in thresholds]

opt_idx = np.argmax(f1_scores)
# Threshold at maximal F1 score
opt_tau = thresholds[opt_idx]
# Accuracy at maximal F1 score
opt_acc = accuracy_score(identical, distances < opt_tau)

# Plot F1 score and accuracy as function of distance threshold
plt.plot(thresholds, f1_scores, label='F1 score')
plt.plot(thresholds, acc_scores, label='Accuracy')
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
plt.title('Accuracy at threshold {:.2f} = {:.3f}'.format(opt_tau, opt_acc))
plt.xlabel('Distance threshold')
plt.legend()

dist_pos = distances[identical == 1]
dist_neg = distances[identical == 0]

plt.figure(figsize=(12,4))

plt.subplot(121)
plt.hist(dist_pos)
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
plt.title('Distances (pos. pairs)')
plt.legend()

plt.subplot(122)
plt.hist(dist_neg)
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
plt.title('Distances (neg. pairs)')
plt.legend()

plt.show()