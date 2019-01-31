import numpy as np
# Plotting
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class Plotting(object):
    """ Purpose: To group all the plotting methods."""

    def plot_margin(self,X1, X2, objFit):
        #-------------------------------------------------------
        # Purpose: Plot the SVM data and margin.
        # Inputs:
        #       X1      :   Vector of X1 data.
        #       X2      :   Vector of X2 data.
        #       objFit  :   The NVM fit object.
        #-------------------------------------------------------

        def f(x, w, b, c=0):
            # -------------------------------------------------------
            # Purpose: Create the margin line given the intercept and coefficients (weights).
            #          Given X1, return X2 such that [X1,X2] in on the line:
            #                       w.X1 + b = c
            # Inputs:
            #       x       :       data
            #       w       :       Coefficient weights.
            #       c       :       Soft margin parameter. c=0 for hard margin.
            # -------------------------------------------------------
            return (-w[0] * x - b + c) / w[1]

        fig = plt.figure()  # create a figure object
        ax = fig.add_subplot(1, 1, 1)
        # Format plot area:
        ax = plt.gca()
        ax = plt.axes(facecolor='#E6E6E6')  # use a gray background.
        ax.set_axisbelow(True)
        plt.grid(color='w', linestyle='solid')  # draw solid white grid lines.
        # hide axis spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        # hide top and right ticks
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()
        # lighten ticks and labels
        ax.tick_params(colors='gray', direction='out')
        for tick in ax.get_xticklabels():
            tick.set_color('gray')
        for tick in ax.get_yticklabels():
            tick.set_color('gray')
        # Format axis.
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))  # Seaprate 000 with ,.
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))  # Seaprate 000 with ,.
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'));  # 2dp for x axis.
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'));  # 2dp for y axis.
        ax.xaxis.set_tick_params(labelsize=8);  # Tick label size.
        ax.yaxis.set_tick_params(labelsize=8);  # Tick label size.
        # Axis limits.
        x1_min, x1_max = X1.min(), X1.max()
        x2_min, x2_max = X2.min(), X2.max()
        ax.set(xlim=(x1_min, x1_max), ylim=(x2_min, x2_max))
        # Labels
        plt.xlabel('$x_1$', fontsize=9)
        plt.ylabel('$x_2$', fontsize=9)
        plt.title('Support Vector Machines - Library used: {0} - Using Kernel: {1}'.format(objFit.fit_type,objFit.kernel), fontsize=9);
        # Set the legend
        legendElements = [
            Line2D([0], [0], linestyle='none',marker='x', color='lightblue',markerfacecolor='lightblue', markersize=9),
            Line2D([0], [0], linestyle='none',marker='o', color='darkorange', markerfacecolor='darkorange', markersize=9),
            Line2D([0], [0], linestyle='-', marker='.', color='black', markerfacecolor='darkorange',markersize=0),
            Line2D([0], [0], linestyle='--', marker='.', color='black', markerfacecolor='darkorange', markersize=0),
            Line2D([0], [0], linestyle='none', marker='.', color='blue', markerfacecolor='blue', markersize=9)
        ];

        if objFit.kernel.upper().find("LINEAR")==-1:
            # Place the legend in a nicer position.
            myLegend = plt.legend(legendElements,['Negative -1', 'Positive +1', 'Decision Boundary', 'Margin', 'Support Vectors'],fontsize="7", shadow=True, loc='lower left', bbox_to_anchor=(0.03, 0.03))
        else:
            myLegend = plt.legend(legendElements,   ['Negative -1', 'Positive +1','Decision Boundary','Margin','Support Vectors'],fontsize="7", shadow=True,loc='top left', bbox_to_anchor=(0.3, 0.98))

        myLegend.get_frame().set_linewidth(0.3)

        # Add the plots:
        plt.plot(X1[:, 0], X1[:, 1], marker='x',markersize=5, color='lightblue',linestyle='none')
        plt.plot(X2[:, 0], X2[:, 1], marker='o',markersize=4, color='darkorange',linestyle='none')
        if objFit.fit_type != 'sklearn':
            plt.scatter(objFit.sv[:, 0], objFit.sv[:, 1], s=60, color="blue")   # The points desginating the support vectors.
        if objFit.fit_type == 'sklearn':
            plt.scatter(objFit.support_vectors_[:, 0], objFit.support_vectors_[:, 1], s=60, color="blue")   # The points desginating the support vectors.

        if objFit.fit_type == 'sklearn' or objFit.kernel  == 'polynomial' or objFit.kernel  == 'gaussian':
            # Non-linear margin line needs to be generated. Will use a contour plot.
            _X1, _X2 = np.meshgrid(np.linspace(x1_min, x1_max, 50), np.linspace(x1_min, x1_max, 50))
            X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(_X1), np.ravel(_X2))])

            if objFit.fit_type == 'sklearn':
                Z = objFit.decision_function(X).reshape(_X1.shape)
            elif objFit.kernel == 'polynomial' or objFit.kernel == 'gaussian':
                Z = objFit.project(X).reshape(_X1.shape)
            else:
                print("unknown fit_type")
                return

            plt.contour(_X1, _X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
            plt.contour(_X1, _X2, Z + 1, [0.0], colors='grey', linestyles='--', linewidths=1, origin='lower')
            plt.contour(_X1, _X2, Z - 1, [0.0], colors='grey', linestyles='--', linewidths=1, origin='lower')
        else:
            # Linear margin line needs to be generated.
            # This section can be done by the above code and use plt.contour. But wanting to generate the linear lines here for demonstration.
            # Decision Boundary:  w.x + b = 0
            _y1 = f(x1_min, objFit.w, objFit.b)
            _y2 = f(x1_max, objFit.w, objFit.b)
            plt.plot([x1_min, x1_max], [_y1, _y2], "k")

            # Margin Upper: w.x + b = 1
            _y3 = f(x1_min, objFit.w, objFit.b, 1)
            _y4 = f(x1_max, objFit.w, objFit.b, 1)
            plt.plot([x1_min, x1_max], [_y3, _y4], "k--")

            # Margin Lower: w.x + b = -1
            _y5 = f(x1_min, objFit.w, objFit.b, -1)
            _y6 = f(x1_max, objFit.w, objFit.b, -1)
            plt.plot([x1_min, x1_max], [_y5, _y6], "k--")

        plt.show(block=False)

