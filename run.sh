for i in 0
do
	python experiments.py --model=vgg-9 \
		--dataset=cifar10 \
		--alg=fedconcat \
		--lr=0.01 \
		--batch-size=64 \
		--epochs=10 \
		--n_parties=40 \
		--rho=0.9 \
		--encoder_round=50 \
		--classifier_round=1000 \
		--n_clusters=5 \
		--partition=noniid-#label2 \
		--beta=0.5\
		--device='cuda:3'\
		--datadir='./data/' \
		--logdir='./logs/' \
		--noise=0\
		--init_seed=$i

	python experiments.py --model=vgg-9 \
		--dataset=cifar10 \
		--alg=fedconcat \
		--lr=0.01 \
		--batch-size=64 \
		--epochs=10 \
		--n_parties=40 \
		--rho=0.9 \
		--encoder_round=50 \
		--classifier_round=1000 \
		--n_clusters=5 \
		--partition=noniid-labeldir \
		--beta=0.5\
		--device='cuda:3'\
		--datadir='./data/' \
		--logdir='./logs/' \
		--noise=0\
		--init_seed=$i
done
