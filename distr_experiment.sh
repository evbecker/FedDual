for deg in 0 1 2 3 
do
	python fed_avg.py \
    		--seed 3 \
    		--tsboard \
    		--dim 10 \
    		--frac_nonzero 0.5 \
    		--noise 0.1 \
            --degree $deg \
    		--save_path 'distribution'

    python fed_dual_avg.py \
    		--seed 3 \
    		--tsboard \
    		--dim 10 \
    		--frac_nonzero 0.5 \
    		--noise 0.1 \
    		--lam 0.01 \
            --degree $deg \
    		--save_path 'distribution'

    python fed_dcd.py \
    		--seed 3 \
    		--tsboard \
    		--dim 10 \
    		--frac_nonzero 0.5 \
    		--noise 0.1 \
            --degree $deg \
    		--save_path 'distribution' 

done