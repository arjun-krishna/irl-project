import pickle
import matplotlib.pyplot as plt
plt.style.use('ggplot')

models = ['D50_bc', 'D50_moco_bc', 'D100_bc', 'D100_moco_bc', 'D200_bc', 'D200_moco_bc', 'expert']

success_rate = []
metric_steps = []

for model in models:
    with open('results/' + model + '.pickle', 'rb') as fp:
        d = pickle.load(fp)
        success_rate.append(d['success_rate'])
        metric_steps.append(d['metric_steps'])

bl = plt.bar(models, success_rate)
for i in range(len(models)):
    if i % 2 == 0:
        bl[i].set_color('#E69F00')
    else:
        bl[i].set_color('#56B4E9')
bl[-1].set_color('#009E73')

plt.ylabel('success rate')
plt.title('Success Rate')
plt.show()

medianprops = dict(color="black",linewidth=1.5)

bp = plt.boxplot(metric_steps, patch_artist=True, medianprops=medianprops)
for i in range(len(models)):
    if i % 2 == 0:
        bp['boxes'][i].set_facecolor('#E69F00')
    else:
        bp['boxes'][i].set_facecolor('#56B4E9')
bp['boxes'][-1].set_facecolor('#009E73')
plt.xticks(range(1, len(models) + 1), models)
plt.ylabel('steps')
plt.title('Steps in an episode\n(averaged over 100 episodes)')
plt.show()