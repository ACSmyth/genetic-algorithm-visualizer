import math
import random
import matplotlib.pyplot as plt
import importlib
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
from nnlib.genetic_algorithm import GeneticAlgorithm


w, h = 700, 700
pygame.init()
pygame.display.set_caption('Neural Network')
screen = pygame.display.set_mode([w, h])




def lerp(x, a, b, c, d):
    return (((d - c) / (b - a)) * (x - a)) + c

# estimate distributions with genetic algorithm

func = lambda x: math.sin(x)
x_min, x_max = 0, 5
y_min, y_max = -1, 1

num_data_pts = 30
xs = []
ys = []
for q in range(num_data_pts):
    x = lerp(q, 0, num_data_pts-1, x_min, x_max)
    xs.append(x)
    ys.append(func(x))


# "game state" is [input, output], NORMALIZED
# just pass in the input
def input_func(state):
    return [state[0]]

def run_game_func(agent, input_func):
    avg_diff = 0
    highest_diff = 0
    for inp in xs:
        inp_normalized = lerp(inp, x_min, x_max, 0, 1)
        raw_out = agent.forward_propagate(input_func([inp_normalized, None]))[0].val
        output_translated = lerp(raw_out, 0, 1, y_min, y_max)
        expected = func(inp)
        highest_diff = max(abs(output_translated - expected), highest_diff)
        avg_diff += abs(output_translated - expected)
    avg_diff /= len(xs) # TODO - eitheravg diff or max of diffs
    return avg_diff

def fitness_func(state):
    return -state



all_xs = []
all_ys = []

def gen_data(agent):
    global all_xs
    global all_ys
    x = x_min
    inc = (x_max - x_min) / 40
    agent_xs = []
    agent_ys = []
    while x <= x_max:
        agent_xs.append(x)
        y = lerp(agent.forward_propagate(input_func([lerp(x, x_min, x_max, 0, 1), None]))[0].val, 0, 1, y_min, y_max)
        agent_ys.append(y)
        x += inc
    all_xs.append(agent_xs)
    all_ys.append(agent_ys)

best_fitness = -999999999
gen = GeneticAlgorithm([1, 10, 5, 1], 100, 1, input_func, run_game_func, fitness_func)
runs = 5000
fitnesses = []
last_injected = 0
for q in range(runs):
    for event in pygame.event.get():
        if event == pygame.QUIT:
            pygame.quit()

    gen.run_generation()
    gen_data(gen.get_best_agent()[0])

    def draw(xs, ys, col, lines=False):
        for z in range(len(xs)):
            x_val = xs[z]
            y_val = ys[z]
            x_draw = int(lerp(x_val, x_min, x_max, 0, w))
            y_draw = int(lerp(y_val, y_min, y_max, h, 0))
            if not lines:
                pygame.draw.circle(screen, col, (x_draw, y_draw), 5)
            if lines and z != 0:
                pygame.draw.line(screen, col, (prev_x, prev_y), (x_draw, y_draw), 3)
            prev_x = x_draw
            prev_y = y_draw

    screen.fill((255, 255, 255))
    xs0 = all_xs[len(all_xs)-1]
    ys0 = all_ys[len(all_xs)-1]

    fit = gen.get_best_agent()[1]
    fitnesses.append(fit)

    #if fit > best_fitness:
    draw(xs, ys, (255, 0, 0))
    draw(xs0, ys0, (0,0,0), lines=True)
    pygame.display.flip()


    last_injected += 1
    if last_injected > 30 and q >= 30 and abs(fitnesses[q-30] - fit) < 0.01:
        last_injected = 0
        # inject mutation
        print('Injecting more mutation')
        for ag in gen.population:
            for z in range(30):
                ag[0].mutate()

    if fit > best_fitness:
        best_fitness = fit
        print('New best fitness: ' + str(1+best_fitness))

pygame.quit()
