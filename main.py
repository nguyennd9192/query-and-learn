

from absl import app
from run_experiment import run
from proc_results import main_proc

def main(argv):
	run() 
	main_proc()
if __name__ == "__main__":
  app.run(main)