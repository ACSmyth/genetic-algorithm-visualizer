import math
import random
import sys
import matplotlib.pyplot as plt
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import importlib
spec = importlib.util.spec_from_file_location("nn", "/Users/alexsmyth/git/neural-network/main.py")
nn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nn)


w, h = 700, 700
pygame.init()
pygame.display.set_caption('Neural Network')
screen = pygame.display.set_mode([w, h])

def lerp(x, a, b, c, d):
    return (((d - c) / (b - a)) * (x - a)) + c

# estimate distributions with genetic algorithm

func = lambda x: math.sin(x**2)
x_min, x_max = 0, 2
y_min, y_max = -1, 1

num_data_pts = 50
xs = []
ys = []
for q in range(num_data_pts):
    x = q * (x_max - x_min) / num_data_pts
    xs.append(x)
    ys.append(func(x))

def fitness(agent):
    avg_diff = 0
    highest_diff = 0
    for inp in xs:
        inp_arr = [0 for q in range(num_inps)]
        inp_idx = int(lerp(inp, x_min, x_max, 0, num_inps))
        inp_arr[inp_idx] = 1


        ys = [n.val for n in agent.forward_propagate(inp_arr)]
        y = ys.index(max(ys))
        y = lerp(y, 0, num_outs, y_min, y_max)

        output_translated = y
        expected = func(inp)
        highest_diff = max(abs(output_translated - expected), highest_diff)
        avg_diff += abs(output_translated - expected)
    avg_diff /= len(xs)
    return -avg_diff

all_xs = []
all_ys = []
fitnesses = []

def gen_data(agent):
    global all_xs
    global all_ys
    global fitnesses
    x = x_min
    inc = (x_max - x_min) / 100
    agent_xs = []
    agent_ys = []
    while x <= x_max:
        agent_xs.append(x)

        inp_arr = [0 for q in range(num_inps)]
        inp_idx = int(lerp(x, x_min, x_max, 0, num_inps))
        inp_arr[inp_idx] = 1

        #y = lerp(agent.forward_propagate([lerp(x, x_min, x_max, 0, 1)])[0].val, 0, 1, y_min, y_max)
        ys = [n.val for n in agent.forward_propagate(inp_arr)]
        y = ys.index(max(ys))
        y = lerp(y, 0, num_outs, y_min, y_max)

        agent_ys.append(y)
        x += inc
    all_xs.append(agent_xs)
    all_ys.append(agent_ys)
    fitnesses.append(fitness(agent))


num_inps = 30
num_outs = 30

agent = nn.NeuralNetwork([num_inps,40,num_outs])
overall_runs = 200
batch_runs = num_data_pts
prev_percent = 0
best_fit = -99999999999
prev_percent = 0
for q in range(overall_runs):
    for z in range(batch_runs):
        raw_inp = xs[z]

        inp = [0 for q in range(num_inps)]
        inp_idx = int(lerp(raw_inp, x_min, x_max, 0, num_inps))
        inp[inp_idx] = 1

        #inp_normalized = lerp(inp, x_min, x_max, 0, 1)
        res = agent.forward_propagate(inp)
        #print([e.val for e in res])

        correct_output_idx = int(lerp(func(raw_inp), y_min, y_max, 0, num_outs))
        correct_output = [0 for q in range(num_outs)]
        correct_output[correct_output_idx] = 1

        agent.back_propagate_queue_weight_changes(correct_output)

    agent.make_weight_changes()
    gen_data(agent)

    # draw data
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.display.quit()
            pygame.quit()
            sys.exit()

    def draw(xs, ys, col):
        for z in range(len(xs)):
            x_val = xs[z]
            y_val = ys[z]
            x_draw = int(lerp(x_val, x_min, x_max, 0, w))
            y_draw = int(lerp(y_val, y_min, y_max, h, 0))
            pygame.draw.circle(screen, col, (x_draw, y_draw), 5)

    screen.fill((255, 255, 255))
    xs0 = all_xs[len(all_xs)-1]
    ys0 = all_ys[len(all_xs)-1]
    draw(xs, ys, (255, 0, 0))
    draw(xs0, ys0, (0,0,0))


    pygame.display.flip()


    # evaluate fitness
    fit = fitness(agent)
    if fit > best_fit:
        best_fit = fit
        print('new best fitness: ' + str(fit))

    cur_percent = int(100 * (q+1) / overall_runs)
    if cur_percent > prev_percent:
        prev_percent = cur_percent
        print(str(cur_percent) + '%')


#plt.scatter(xs, ys, 100)
best_idx = fitnesses.index(max(fitnesses))
best_fit_xs = all_xs[best_idx]
best_fit_ys = all_ys[best_idx]
#plt.scatter(best_fit_xs, best_fit_ys)

#for q in range(len(all_xs)):
#   agent_xs = all_xs[q]
#   agent_ys = all_ys[q]
#   plt.scatter(agent_xs, agent_ys)

plt.show()
pygame.display.quit()
pygame.quit()







