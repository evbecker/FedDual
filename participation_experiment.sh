for frac in 0.1 0.3 0.5 0.7 0.9 
do
	python fed_avg.py \
    		--seed 3 \
    		--tsboard \
    		--dim 10 \
    		--frac_nonzero 0.5 \
    		--noise 0.1 \
            --frac $frac \
    		--save_path 'participation'

    python fed_dual_avg.py \
    		--seed 3 \
    		--tsboard \
    		--dim 10 \
    		--frac_nonzero 0.5 \
    		--noise 0.1 \
    		--lam 0.01 \
            --frac $frac \
    		--save_path 'participation'

    python fed_dcd.py \
    		--seed 3 \
    		--tsboard \
    		--dim 10 \
    		--frac_nonzero 0.5 \
    		--noise 0.1 \
            --frac $frac \
    		--save_path 'participation' 

done