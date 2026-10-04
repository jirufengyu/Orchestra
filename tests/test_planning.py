import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from embodied_agent.contracts import Goal
from embodied_agent.executive import OpenAIChatModel
from embodied_agent.planning import LlmTaskPlanner
from embodied_agent.prompts import TASK_AGENT_SYSTEM_PROMPT


class PlanningTests(unittest.TestCase):
    def test_paper_array_preserves_order_repetitions_and_entities(self):
        step = {"subtask": "Move the apple", "activate object": ["red apple"]}
        model = SimpleNamespace(complete_json=Mock(return_value=[step, step]))
        planner = LlmTaskPlanner(
            model,
            atomic_goal_resolver=lambda text, i: Goal(
                str(i), text, {"A": "1"}, {"task_id": "fruit"}
            ),
            skill_context={"available_skills": ["move"]},
        )
        goals = planner.plan("Move the apple twice")
        self.assertEqual([g.instruction for g in goals], ["Move the apple"] * 2)
        self.assertEqual(goals[0].metadata["activate object"], ["red apple"])
        self.assertEqual(goals[0].metadata["task_id"], "fruit")
        self.assertEqual(goals[0].entity_bindings, {"A": "1"})
        prompt, payload = model.complete_json.call_args.args
        self.assertEqual(prompt, TASK_AGENT_SYSTEM_PROMPT)
        self.assertEqual(json.loads(payload)["registered_action_policy_capabilities"], ["move"])

    def test_legacy_object_and_short_mode(self):
        model = SimpleNamespace(complete_json=Mock(return_value={
            "subtasks": [{"instruction": "Move the apple"}]
        }))
        planner = LlmTaskPlanner(model)
        self.assertEqual(planner.plan("Move the apple")[0].instruction, "Move the apple")
        model.complete_json.reset_mock()
        self.assertEqual(planner.plan("One step", mode="short")[0].instruction, "One step")
        model.complete_json.assert_not_called()

    def test_invalid_plans(self):
        for payload in [[], None, 42, [{"activate object": []}],
                        [{"subtask": "Move", "activate object": "apple"}]]:
            with self.subTest(payload=payload):
                model = SimpleNamespace(complete_json=Mock(return_value=payload))
                with self.assertRaises(ValueError):
                    LlmTaskPlanner(model).plan("Move")

    def test_array_transport_does_not_request_json_object(self):
        model = OpenAIChatModel.__new__(OpenAIChatModel)
        model._model = "test"
        create = Mock(return_value=SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=json.dumps([
                {"subtask": "Move", "activate object": ["apple"]}
            ])))
        ]))
        model._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        goals = LlmTaskPlanner(model).plan("Move")
        self.assertEqual(goals[0].instruction, "Move")
        self.assertNotIn("response_format", create.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
