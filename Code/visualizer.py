import matplotlib.pyplot as plt
from constants import generated_files_path

def draw_multiple_line_plots(dict_of_vals, field_to_draw, title, y_label, x_label, file_save_name):
    legend_li = []
    for k in dict_of_vals.keys():
        plt.plot(dict_of_vals[k].history[field_to_draw])
        legend_li.append(k)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(legend_li, loc='upper left')
    plt.savefig(generated_files_path+file_save_name)
    plt.cla()