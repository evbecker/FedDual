
for noise in 0.01 0.1 1
do
	python fed_avg.py \
    		--seed 3 \
    		--tsboard \
    		--dim 10 \
    		--frac_nonzero 0.5 \
    		--noise $noise \
    		--save_path 'noise'

    python fed_dual_avg.py \
    		--seed 3 \
    		--tsboard \
    		--dim 10 \
    		--frac_nonzero 0.5 \
    		--noise $noise \
    		--lam 0.01 \
    		--save_path 'noise'

    python fed_dcd.py \
    		--seed 3 \
    		--tsboard \
    		--dim 10 \
    		--frac_nonzero 0.5 \
    		--noise $noise \
    		--save_path 'noise' 

done
