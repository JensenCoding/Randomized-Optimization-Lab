import numpy as np
import matplotlib.pyplot as plt
import mlrose
import time
import os


class Optima:
    COLOREDGE = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4), (1, 2), (2, 4), (1, 5), (1, 6), (1, 4), (3, 4), (4, 5),
                 (2, 3), (2, 4), (2, 6), (3, 5), (4, 2), (4, 5), (5, 6), (8, 9), (7, 6), (3, 9), (2, 7), (4, 1), (3, 8), (4, 1),
                 (1, 2), (1, 3), (1, 5), (2, 4), (3, 1), (3, 4), (4, 5), (2, 3), (3, 5), (2, 6), (2, 7), (2, 5), (4, 5), (5, 6),
                 (3, 4), (3, 5), (3, 7), (4, 6), (5, 3), (5, 6), (6, 7), (9, 10), (8, 7), (4, 10), (3, 8), (5, 2), (4, 9), (5, 2),
                 (2, 4), (2, 5), (2, 7), (3, 6), (4, 3), (4, 6), (5, 7), (3, 5), (4, 7), (3, 8), (3, 9), (3, 7), (5, 7), (6, 8),
                 (4, 6), (4, 7), (4, 9), (5, 8), (6, 5), (6, 8), (7, 9), (10, 12), (9, 9), (5, 12), (4, 10), (6, 4), (5, 11),
                 (6, 4), (3, 5), (3, 6), (3, 8), (4, 7), (5, 4), (5, 7), (6, 8), (4, 6), (5, 8), (4, 9), (4, 10), (4, 8), (6, 8),
                 (7, 9), (5, 7), (5, 8), (5, 10), (6, 9), (7, 6), (7, 9), (8, 10), (11, 13), (10, 10), (6, 13), (5, 11), (7, 5),
                 (6, 12), (7, 5)]

    def __init__(self, prob_name, output_path):
        np.random.seed(7641)
        self.prob_name = prob_name
        self.output_path = output_path

        self.problem = None
        self.init_state = None
        self.schedule = None
        self.restarts = None
        self.mutation_prob = None
        self.keep_pct = None
        self.pop_size = None

    def load_data(self):
        pass

    def timing_curve(self):
        print("Creating timing curve")
        fitness_sa_arr = []
        fitness_rhc_arr = []
        fitness_ga_arr = []
        fitness_mimic_arr = []
        time_sa_arr = []
        time_rhc_arr = []
        time_ga_arr = []
        time_mimic_arr = []

        n_range = range(15, 100, 20)

        for n in n_range:
            problem, init_state = self.get_prob(t_pct=0.15, p_length=n)

            st = time.time()
            _, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, schedule=self.schedule, max_attempts=100, max_iters=5000, init_state=init_state, curve=True)
            end = time.time()
            sa_time = end - st

            st = time.time()
            _, best_fitness_rhc, fitness_curve_rhc = mlrose.random_hill_climb(problem,  max_attempts=100, max_iters=5000, init_state=init_state, curve=True)
            end = time.time()
            rhc_time = end - st

            st = time.time()
            _, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem, mutation_prob=self.mutation_prob, pop_size=self.pop_size, max_attempts=100, max_iters=5000, curve=True)
            end = time.time()
            ga_time = end - st

            st = time.time()
            _, best_fitness_mimic, fitness_curve_mimic = mlrose.mimic(problem, keep_pct=self.keep_pct, pop_size=self.pop_size, max_attempts=100, max_iters=5000, curve=True)
            end = time.time()
            mimic_time = end - st

            fitness_sa_arr.append(best_fitness_sa)
            fitness_rhc_arr.append(best_fitness_rhc)
            fitness_ga_arr.append(best_fitness_ga)
            fitness_mimic_arr.append(best_fitness_mimic)

            time_sa_arr.append(sa_time)
            time_rhc_arr.append(rhc_time)
            time_ga_arr.append(ga_time)
            time_mimic_arr.append(mimic_time)

        plt.close()
        plt.figure()
        # plt.plot(n_range, np.array(fitness_rhc_arr), label='RHC')
        plt.plot(n_range, np.array(fitness_sa_arr), label='SA')
        plt.plot(n_range, np.array(fitness_ga_arr), label='GA')
        plt.plot(n_range, np.array(fitness_mimic_arr), label='MIMIC')
        plt.xlabel('Input Size')
        plt.ylabel('Fitness')
        plt.legend(loc="best")
        plt.title("{}: Fitness vs. Input Size".format(self.prob_name))
        plt.savefig(os.path.join(self.output_path, "{}_size_fitness.png".format(self.prob_name)))

        plt.close()
        plt.figure()
        # plt.plot(n_range, np.array(time_rhc_arr), label='RHC')
        plt.plot(n_range, np.array(time_sa_arr), label='SA')
        plt.plot(n_range, np.array(time_ga_arr), label='GA')
        plt.plot(n_range, np.array(time_mimic_arr), label='MIMIC')
        plt.legend(loc="best")
        plt.xlabel('Input Size')
        plt.ylabel('Time (s)')
        plt.title("{}: Time vs. Input Size".format(self.prob_name))
        plt.savefig(os.path.join(self.output_path, "{}_size_timing.png".format(self.prob_name)))

    def iteration_curve(self):
        print("Creating iteration curve")
        problem, init_state = self.get_prob(t_pct=0.15)
        best_state_rhc, best_fitness_rhc, fitness_curve_rhc = mlrose.random_hill_climb(problem, max_attempts=1000, max_iters=5000, init_state=init_state, curve=True)
        best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, max_attempts=10000, max_iters=5000, init_state=init_state, curve=True)
        problem, init_state = self.get_prob(t_pct=0.15)
        best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem, max_attempts=1000, max_iters=5000, curve=True)
        problem, init_state = self.get_prob(t_pct=0.15)
        best_state_mimic, best_fitness_mimic, fitness_curve_mimic = mlrose.mimic(problem, pop_size=self.pop_size, max_attempts=100, max_iters=5000, curve=True)

        plt.figure()
        plt.plot(fitness_curve_rhc, label='RHC')
        plt.plot(fitness_curve_sa, label='SA')
        plt.plot(fitness_curve_ga, label='GA')
        plt.plot(fitness_curve_mimic, label='MIMIC')
        plt.legend(loc="best")
        plt.ylabel('Fitness')
        plt.xlabel('Number of Iterations')
        plt.title("{}: Fitness vs. Number of Iterations".format(self.prob_name))
        # ax = plt.gca()
        # ax.set_xscale('log')
        plt.savefig(os.path.join(self.output_path, "{}_iterations.png".format(self.prob_name)))

    def get_prob(self, t_pct=None, p_length=None):
        if self.prob_name == 'Four Peaks':
            fitness = mlrose.FourPeaks(t_pct)
            p_len = 100
            self.schedule = mlrose.ExpDecay()
            self.restarts = 0
            self.mutation_prob = 0.1
            self.keep_pct = 0.1
            self.pop_size = 500
        elif self.prob_name == "Continuous Peaks":
            fitness = mlrose.ContinuousPeaks(t_pct)
            p_len = 100
            self.schedule = mlrose.GeomDecay()
            self.restarts = 0
            self.mutation_prob = 0.1
            self.keep_pct = 0.2
            self.pop_size = 200
        elif self.prob_name == "Max K Color":
            fitness = mlrose.MaxKColor(self.COLOREDGE)
            p_len = 100
            self.schedule = mlrose.ExpDecay()
            self.restarts = 0
            self.mutation_prob = 0.2
            self.keep_pct = 0.2
            self.pop_size = 200
        elif self.prob_name == "Flip Flop":
            fitness = mlrose.FlipFlop()
            p_len = 100
            self.schedule = mlrose.ArithDecay()
            self.restarts = 0
            self.mutation_prob = 0.2
            self.keep_pct = 0.5
            self.pop_size = 500
        elif self.prob_name == "One Max":
            fitness = mlrose.OneMax()
            p_len = 100
            self.schedule = mlrose.GeomDecay()
            self.restarts = 0
            self.mutation_prob = 0.2
            self.keep_pct = 0.1
            self.pop_size = 100
        else:
            fitness = None
            p_len = 0

        if p_length is None:
            p_length = p_len

        problem = mlrose.DiscreteOpt(length=p_length, fitness_fn=fitness)
        init_state = np.random.randint(2, size=p_length)
        return problem, init_state

    def rhc(self):
        print("Creating RHC curve")
        problem, init_state = self.get_prob(t_pct=0.15)
        plt.close()
        plt.figure()
        for restarts in range(0, 11, 2):
            _, _, fitness_curve = mlrose.random_hill_climb(problem, restarts=restarts, max_attempts=100, max_iters=1000, init_state=init_state, curve=True)
            plt.plot(fitness_curve, label="restarts={}".format(restarts))

        plt.title("{}: Randomized Hill Climbing".format(self.prob_name))
        plt.legend(loc="best")
        plt.xlabel('Number of Iterations')
        plt.ylabel('Fitness')
        plt.savefig(os.path.join(self.output_path, "{}_RHC Analysis.png".format(self.prob_name)))

    def sa(self):
        print("Creating SA curve")
        problem, init_state = self.get_prob(t_pct=0.15)
        plt.close()
        plt.figure()
        for schedule, s_str in zip([mlrose.GeomDecay(), mlrose.ArithDecay(), mlrose.ExpDecay()], ['GeomDecay', 'ArithDecay', 'ExpDecay']):
            _, _, fitness_curve = mlrose.simulated_annealing(problem, schedule=schedule, max_attempts=100, max_iters=5000, init_state=init_state, curve=True)
            plt.plot(fitness_curve, label="schedule={}".format(s_str))

        plt.title("{}: Simulated Annealing".format(self.prob_name))
        plt.legend(loc="best")
        plt.xlabel('Number of Iterations')
        plt.ylabel('Fitness')
        plt.savefig(os.path.join(self.output_path, "{}_SA Analysis.png".format(self.prob_name)))

    def ga(self):
        print("Creating GA curve")
        problem, _ = self.get_prob(t_pct=0.15)
        plt.close()
        plt.figure()
        for pop_size in [100, 200, 500]:
            for mutation_prob in [0.1, 0.2, 0.5]:
                _, _, fitness_curve = mlrose.genetic_alg(problem, mutation_prob=mutation_prob, pop_size=pop_size, max_attempts=100, max_iters=5000, curve=True)
                plt.plot(fitness_curve, label="mutation_prob={}, pop_size={}".format(mutation_prob, pop_size))

        plt.title("{}: Genetic Algorithm".format(self.prob_name))
        plt.legend(loc="best")
        plt.xlabel('Number of Iterations')
        plt.ylabel('Fitness')
        plt.savefig(os.path.join(self.output_path, "{}_GA Analysis.png".format(self.prob_name)))

    def mimic(self):
        print("Creating Mimic curve")
        problem, _ = self.get_prob(t_pct=0.15)
        plt.close()
        plt.figure()
        for pop_size in [100, 200, 500]:
            for keep_pct in [0.1, 0.2, 0.5]:
                _, _, fitness_curve = mlrose.mimic(problem, keep_pct=keep_pct, pop_size=pop_size, max_attempts=100, max_iters=5000, curve=True)
                plt.plot(fitness_curve, label="keep_pct={}, pop_size={}".format(keep_pct, pop_size))

        plt.title("{}: MIMIC".format(self.prob_name))
        plt.legend(loc="best")
        plt.xlabel('Number of Iterations')
        plt.ylabel('Fitness')
        plt.savefig(os.path.join(self.output_path, "{}_MIMIC Analysis.png".format(self.prob_name)))
