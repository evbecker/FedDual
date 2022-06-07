
for sparsity in 0.25 0.5 1
do
	python fed_avg.py \
    		--seed 3 \
    		--tsboard \
    		--dim 10 \
    		--frac_nonzero $sparsity \
    		--noise 0.1 \
    		--save_path 'sparsity'

    python fed_dual_avg.py \
    		--seed 3 \
    		--tsboard \
    		--dim 10 \
    		--frac_nonzero $sparsity \
    		--noise 0.1 \
    		--lam 0.01 \
    		--save_path 'sparsity'

    python fed_dcd.py \
    		--seed 3 \
    		--tsboard \
    		--dim 10 \
    		--frac_nonzero $sparsity \
    		--noise 0.1 \
    		--save_path 'sparsity' 

done
