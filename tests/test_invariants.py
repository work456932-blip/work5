"""Offline correctness checks; no datasets, model weights or API calls."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import tempfile
import unittest
from scenerepair.config import Config
from scenerepair.schema import State,Trajectory,Example
from scenerepair.graph import DAG,build_graphs
from scenerepair.math_utils import js_divergence,lcb,necessity,sufficiency,root_score
from scenerepair.repair import regenerate
from scenerepair.data import load_examples
from scenerepair.diagnosis import select_root

def state(i,typ,parents):
    return State(i,typ,"A typed commitment",["chair"],"camera",None,[],[],{},["v0"],parents,.5)

class Invariants(unittest.TestCase):
    def test_equations(self):
        p,q = {"A":1.,"B":0.},{"A":0.,"B":1.}
        self.assertAlmostEqual(js_divergence(p,p),0.)
        self.assertAlmostEqual(js_divergence(p,q),math.log(2))
        self.assertAlmostEqual(necessity(p,p,.8,.3,[.5,.5]),.25)
        self.assertAlmostEqual(sufficiency(p,p,.3,.9,.2,.8,[1/3]*3),.4)
        self.assertAlmostEqual(root_score([.5]*4,[1]*4,0),.5)
        self.assertAlmostEqual(lcb([.1,.1,.1],.9),.1)
        self.assertLess(lcb([0.,.2,.4],.9),.2)

    def test_inferred_edges_do_not_expand_edits(self):
        trace = Trajectory([state(0,"correspondence",[]),state(1,"reference_frame",[]),
                            state(2,"counterfactual_operation",[1]),state(3,"answer_mapping",[2])],{"A":.5,"B":.5})
        declared,diagnostic,audit = build_graphs(trace,lambda a,b:True)
        self.assertEqual(declared.descendants(0),set())
        self.assertEqual(diagnostic.descendants(0),{1,2,3})
        self.assertFalse(declared.add(3,1))

    def test_ancestor_competition_and_ambiguity(self):
        graph = DAG([0,1,2],{(0,1)})
        def signal():
            return {"A":.6,"N":.5,"U":.5,"Q":.8,"critic_agreement":True,"interventions":[{"valid":True}]}
        signals = {0:signal(),1:signal()}
        root,audit = select_root(signals,graph,Config())
        self.assertEqual(root,0)
        signals[2]=signal()
        root,audit = select_root(signals,graph,Config())
        self.assertIsNone(root)
        self.assertEqual(audit["status"],"abstain_ambiguous_roots")

    def test_repair_boundary(self):
        original = Trajectory([state(0,"observation",[]),state(1,"reference_frame",[0]),
                               state(2,"answer_mapping",[1])],{"A":.5,"B":.5})
        ex = Example("x","question",{"v0":"unused.png"},{"A":"a","B":"b"})
        edited = original.copy()
        edited.states[1].description="changed frame"
        class Session:
            def ask(self,*args):
                return {"states":[asdict(s) for s in edited.states[1:]],"answer_distribution":{"A":.8,"B":.2},"target_box":None,"target_view":None}
        result = regenerate(Session(),ex,original,{1,2})
        self.assertEqual(asdict(result.states[0]),asdict(original.states[0]))
        self.assertEqual(result.states[1].description,"changed frame")
        with self.assertRaises(ValueError):
            regenerate(Session(),ex,original,{1})

    def test_label_boundary(self):
        record = {"id":"x","question":"q","images":{},"options":{"A":"a","B":"b"},"group":"L4"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"data.jsonl"
            path.write_text(json.dumps({**record,"answer":"A","label":0})+"\n")
            first = list(load_examples(path))[0].public()
            path.write_text(json.dumps({**record,"answer":"B","label":1,"private_annotation":{"root":2}})+"\n")
            second = list(load_examples(path))[0].public()
            self.assertEqual(first,second)

if __name__=="__main__":
    unittest.main()
