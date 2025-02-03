from autorag.deploy import Runner
import pandas as pd
from typing import List, Dict, Optional, Union

class AdvancedRunner(Runner):
    def __init__(self, config: Dict, project_dir: Optional[str] = None):
        super().__init__(config, project_dir)

    def run_with_qa(self, qa_data: pd.DataFrame):
        """
        Run the pipeline with qa_data.
        The loaded pipeline must start with a single query,
        so the first module of the pipeline must be `query_expansion` or `retrieval` module.

        qa_data: pd.DataFrame(
            {
                "qid": str(uuid.uuid4()),
                "query": [query],
                "retrieval_gt": [[]],
                "generation_gt": [""],
            }
        )
        :return: The result of the pipeline.
        """
        assert qa_data is not None, "qa_data must not be None"
        previous_result = qa_data
        for module_instance, module_param in zip(
                self.module_instances, self.module_params
        ):
            print(module_instance, module_param)
            new_result = module_instance.pure(
                previous_result=previous_result, **module_param
            )
            duplicated_columns = previous_result.columns.intersection(
                new_result.columns
            )
            drop_previous_result = previous_result.drop(columns=duplicated_columns)
            previous_result = pd.concat([drop_previous_result, new_result], axis=1)

        return previous_result