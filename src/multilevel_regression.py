from csv import reader as csv_reader
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

mouse_data_files = [
    '../experiments/PyMT_Final/mouse_1_Day5/mouse_1_Day5.txt',
    '../experiments/PyMT_Final/mouse_2_Day5/mouse_2_Day5.txt',
    '../experiments/PyMT_Final/mouse_3_Day5/mouse_3_Day5.txt']

mouse_data_files = [
    '../experiments/C31T_Final/mouse_1_Day5/mouse_1_Day5.txt',
    '../experiments/C31T_Final/mouse_2_Day5/mouse_2_Day5.txt',
    '../experiments/C31T_Final/mouse_3_Day5/mouse_3_Day5.txt',
    '../experiments/C31T_Final/mouse_4_Day5/mouse_4_Day5.txt']

def parse_data_summary(file_location):
    data_table = []
    with open(file_location, 'rb') as source:
        reader = csv_reader(source, delimiter='\t')
        header = reader.next()
        for line in reader:
            data_table.append(line)
    return np.array(data_table).astype(np.float)


def analysis_loop(data_table):

    def perform_regression(col_1, col_2):
        slope, intercept, r_value, p_value, std_err = stats.linregress(col_1, col_2)

        return r_value**2, p_value

    headers = ['ID',
               'Invasion',
               'Size',
               'K14 Peripheral Pixels Sum',
               'K14 Central Pixels Sum',
               'K14 Entire Sum',
               'K14 Entire Mean']

    r2_matrix = np.zeros((5, 5))

    exp_var_matrix = np.zeros((5, 5))


    base_var = np.var(data_table[:, 1])

    for i in range(2, 7):
        slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = stats.linregress(data_table[:, i],
                                                                       data_table[:, 1])
        remaining_variance = data_table[:, 1] - intercept_1 - data_table[:, i]*slope_1
        explained_variance = (1 - np.var(remaining_variance) / base_var) * 100
        print '\t%s: r2: %.2f, p_val: %.2f, rem var: %.2f %%' % (headers[i],
                                                              r_value_1**2,
                                                              p_value_1,
                                                              explained_variance)

        r2_matrix[i-2, i-2] = r_value_1**2
        exp_var_matrix[i-2, i-2] = explained_variance

        for j in range(2, 7):
            if i != j:
                slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = stats.linregress(
                    data_table[:, j],
                    remaining_variance)
                remaining_variance_2 = remaining_variance - intercept_2 - data_table[:, j]*slope_2
                explained_variance_2 = (1 - np.var(remaining_variance_2) / base_var) * 100
                print '\t\t%s: r2: %.2f, p_val: %.2f, rem var: %.2f %%' % (headers[j],
                                                                        r_value_2**2,
                                                                        p_value_2,
                                                                        explained_variance_2)

                r2_matrix[j-2, i-2] = r_value_2**2
                r2_matrix[i-2, j-2] = r_value_2**2

                exp_var_matrix[j-2, i-2] = explained_variance_2
                exp_var_matrix[i-2, j-2] = explained_variance_2

                plt.subplot(211)

                plt.title('%s: r2: %.2f, p_val: %.2f, rem var: %.2f %%' % (headers[i],
                                                              r_value_1**2,
                                                              p_value_1,
                                                              explained_variance))

                plt.plot(data_table[:, i], data_table[:, 1], 'ko')
                plt.plot(data_table[:, i], intercept_1 + slope_1*data_table[:, i], 'r-')

                plt.xlabel(headers[i])
                plt.ylabel(headers[1])

                plt.subplot(212)

                plt.title('%s: r2: %.2f, p_val: %.2f, rem var: %.2f %%' % (headers[j],
                                                                        r_value_2**2,
                                                                        p_value_2,
                                                                        explained_variance_2))

                plt.plot(data_table[:, j], remaining_variance, 'ko')
                plt.plot(data_table[:, j], intercept_2 + slope_2*data_table[:, j], 'r-')

                plt.xlabel(headers[j])
                plt.ylabel('yet unexplained variance')

                # plt.show()

                plt.figure(figsize=(4, 4), dpi=80)
                plt.plot(data_table[:, i], data_table[:, 1], 'ko')
                plt.xlabel(headers[i], fontsize=16)
                plt.ylabel(headers[1], fontsize=16)

                plt.tight_layout()
                # plt.show()

    np.set_printoptions(precision=2, suppress=True)
    print r2_matrix

    print exp_var_matrix

if __name__ == "__main__":
    data_tables = [parse_data_summary(file_location) for file_location in mouse_data_files]

    for table in data_tables:
        print '>'
        analysis_loop(table)
