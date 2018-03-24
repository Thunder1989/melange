import os

def exec_measurement(func):
    def wrapped(*args, **kwargs):
        begin_time = arrow.get()
        res = func(*args, **kwargs)
        end_time = arrow.get()
        print('Execution Time: {0}'.format(end_time - begin_time))
        return res
    return wrapped


class Inferencer(object):
    """
    # input parameters
    - target_building (str): name of the target building. this can be arbitrary later
    - source_buildings (list(str)): list of buildings already known.
    - exp_id: just for logging
    - conf: dictionary of other configuration parameters.
    """

    def __init__(self,
                 target_building,
                 target_srcids=set(),
                 source_buildings=[],
                 exp_id=0,
                 framework_name=None,
                 config={},
                 ):
        super(Inferencer, self).__init__()
        self.exp_id = exp_id # an identifier for logging/debugging
        self.framework_name = framework_name # e.g., Scarbble
        self.config = config # future usage
        #self.infer_g = rdflib.Graph() # future usage
        self.training_srcids = target_srcids # already known srcids
        #self.all_point_tagsets = point_tagsets # all the possible point tagsets
                                               # defined in Brick.
        self.pred = {  # predicted results
            'tagsets': dict(),
            'point': dict(),
            }
        self.target_srcids = target_srcids
        self.history = [] # logging and visualization purpose
        self.required_label_types = ['point', 'fullparsing'] # Future purpose


    def evaluate_points(self):
        curr_log = {
            'training_srcids': self.training_srcids
        }
        score = 0
        for srcid, pred_tagsets in self.pred['tagsets'].items():
            true_tagsets = LabeledMetadata.objects(srcid=srcid)[0].tagsets
            true_point = sel_point_tagset(true_tagsets)
            pred_point = sel_point_tagset(pred_tagsets)
            if true_point == pred_point:
                score +=1
        curr_log['accuracy'] = score / len(self.pred['point'])
        return curr_log


    def evaluate(self):
        points_log = self.evaluate_points()
        log = {
            'points': points_log
        }
        self.history.append(log)


    def plot_result_point(self):
        srcid_nums = [len(log['points']['learned_srcids']) for log in self.history]
        accs = [log['points']['accuracy'] for log in self.history]
        fig, _ = plotter.plot_multiple_2dline(srcid_nums, [accs])
        for ax in fig.axes:
            ax.set_grid(True)
        plot_name = '{0}_points_{1}.pdf'.format(self.framework_name, self.exp_id)
        plotter.save_fig(fig, plot_name)


    # ESSENTIAL
    def run_auto(self, iter_num=1):
        """Learn from the scratch to the end.

        This executes the learning mechanism from the ground truth.
        It iterates for the given amount of the number.
        Basic procedure is iterating this:
            ```python
            f = Framework()
            while not final_condition:
                new_srcids = f.select_informative_samples(10)
                f.update_model(new_srcids)
                self.pred['tagsets'] = XXX
                f.evaluate()
                final_condition = f.get_final_condition()
            ```

        Args:
            iter_num (int): total iteration number.

        Returns:
            None

        Byproduct:


        """
        pass


    # ESSENTIAL
    def update_model(self, srcids):
        """Update model with given newly added srcids.

        This update the model based on the newly added srcids.
        Relevant data for the srcids are given from the ground truth data.
        We can later add an interactive function for users to manually add them

        Args:
            srcids (list(str)): The model will be updated based on the given
                                srcids.
        Returns:
            None

        Byproduct:
            The model will be updated, which can be used for predictions.
        """

        # Get examples from the user if labels do not exist
        for srcid in srcids:
            labeled = LabeledMetadata.objects(srcid=srcid)
            if not labeled:
                # TODO: Add function to receive it from actual user.
                pass


    # ESSENTIAL
    def select_example(self, example_num):
        """Select the most informative N samples from the unlabeled data.

        This function is mainly used by active function frameworks to select
        the most informative samples to ask to the domain experts.
        The chosen samples are again fed back to the model updating mechanisms.

        Args:
            sample_num (int): The number of samples to be chosen

        Returns:
            new_srcids (list(str)): The list of srcids.

        Byproducts:
            None
        """
        pass


    # ESSENTIAL
    def predict(self, target_srcids):
        # TODO
        """
        """
        pass


