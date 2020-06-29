

from absl import app
from run_experiment import run
from proc_results import main_proc

def main(argv):
	extend_save_idx = run() 
	main_proc(extend_save_idx)
if __name__ == "__main__":
  app.run(main)