from ensemble import fit_models, build_ensemble, initialize_hp, initialize_dt, hp_search, greedy_hp, optimize_weights
from numpy import ones
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=4000, n_features=20, n_informative=2, n_redundant=2)

hp = initialize_hp()

dt = initialize_dt(X, y)

print "beginning greedy hp search"

hp = greedy_hp(hp, dt)

print "fitting the models"

rs = fit_models(hp, dt)

weights = ones(hp.num_models)/hp.num_models

en = build_ensemble(rs, weights, max_iter=1000)

print "optimizing weights"

output_list = hp_search(hp_init=hp, rs_init=rs, en_init=en, dt=dt, weights=weights, K=50, max_iter=1000)

tracker = output_list[0]
hp_best = output_list[1]
rs_best = output_list[2]
en_best = output_list[3]

output_list2 = optimize_weights(en=en_best, rs=rs_best)

tracker2 = output_list2[0]
en_best = output_list2[1]
weights_best = output_list2[2]

print "you are awesome"
