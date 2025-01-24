from .base import *



GENERATOR_STATIC_PARAMS = {
    "similarity_threshold_cutoff": { # must start with a embedding model
        "batch": 64
    },
    "similarity_percentile_cutoff": {# must start with a embedding model
        "batch": 64
    },
    # '''Filter out the contents that are below the threshold datetime. If all contents are filtered, keep the only one recency content. If the threshold date format is incorrect, return the original contents.
    # It is useful when you want to use the latest information. The time can be extracted from the corpus metadata.'''
    "recency_filter": {
        "threshold_datetime": "2015-01-01"  # threshold_datetime format should be one of the following three!YYYY-MM-DD YYYY-MM-DD HH:MM YYYY-MM-DD HH:MM:SS
    },
    "threshold_cutoff": {
        "reverse": False #  If True, the lower the score, the better. Default is False.
    },
    "percentile_cutoff": {
        "reverse": False
    },

}


class PassageFilterConfiguration(BaseConfiguration):

    def __init__(self, config: Dict, project_dir: Optional[str] = None):
        super().__init__(config, project_dir)
        self.cs = self.build(config)
        self.para = self.translate(config)

    def build(self, config=None) -> ConfigurationSpace | None:
        cs = ConfigurationSpace(
            space={
                "filter_name": CategoricalHyperparameter("filter_name",
                                                            ["similarity_threshold_cutoff", "similarity_percentile_cutoff",  "recency_filter","threshold_cutoff", "percentile_cutoff"],
                                                            weights=[1, 1, 1, 1, 1]),
            }
        )
        # Add general hyperparameters
        metrics = CategoricalHyperparameter("metrics", choices=[ "retrieval_f1", "retrieval_recall", "retrieval_precision" ])
        threshold = UniformFloatHyperparameter( "threshold", lower=0, upper=1, default_value=0.85)
        percentile = UniformFloatHyperparameter("percentile", lower=0, upper=1, default_value=0.6)
        cs.add([threshold, percentile, metrics])
        cs.add(InCondition(cs["threshold"], cs["filter_name"], ["similarity_threshold_cutoff", "threshold_cutoff"])) # only similarity_threshold_cutoff, threshold_cutoff has the threshold
        cs.add(InCondition(cs["percentile"], cs["filter_name"], ["similarity_percentile_cutoff", "percentile_cutoff"])) # only similarity_percentile_cutoff, percentile_cutoff has the percentile
        return cs

    def sampling(self, size: Optional[int] = 1) -> Union[Configuration, List[Configuration]]:
        return self.cs.sample_configuration(size)

    def translate(self, config) -> Dict:
        return self.cs.sample_configuration(size)
