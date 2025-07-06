source /etc/network_turbo
python3 -m venv venv
git clone https://github.com/tjn-Tommy/LOCC-VQE-reproduction.git
source venv/bin/activate
cd LOCC-VQE-reproduction
pip install -r requirements.txt
pip install -U "jax[cuda12]"
