"""Fast field-study contract tests; no atomic models are constructed."""
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import scan_forster_p0_4_field as field


class FieldStudyTests(unittest.TestCase):
    def test_explicit_geometry(self):
        self.assertEqual(field.coordinates(.05, -1), (3.35, 0.))
        self.assertAlmostEqual(field.coordinates(.05, 1)[0], 3.45)
        self.assertGreater(field.coordinates(.05, 0)[1], 0)

    def test_selected_pulse_and_reference(self):
        np.testing.assert_array_equal(field.PARAMETERS, field.minimax.SELECTED_PARAMETERS)
        self.assertEqual(field.asdict(field.p0.REFERENCE_BASIS), {
            'delta_n': 3, 'atomic_window_ghz': 80., 'pair_window_ghz': 40.,
            'delta_l': 2, 'interaction_order': 4})

    def test_robust_summary(self):
        rows = [dict(radius_um=r, cosine=c, distance_um=3.4, theta_deg=0.,
            position=.999, vertices=[[.998, .997], [.996, .995]])
            for r, c in ((0., 0.), (.05, -1.), (.05, 1.))]
        result = field.summarize(rows, .999)
        self.assertEqual(result['joint_minimum'], .995)
        self.assertEqual(result['worst']['yb_scale'], 1.01)
        self.assertAlmostEqual(result['endpoint_objective'],
            .005 + .02 * (2*(.002+.003+.004+.005)+.001)/9)

    def test_field_passed_to_builder_without_globals(self):
        study = object.__new__(field.Study)
        study.cache = {}
        with patch.object(field, 'build_pair_model', side_effect=RuntimeError('sentinel')) as build:
            with self.assertRaisesRegex(RuntimeError, 'sentinel'):
                study.modes(2.75, .05, -1., 60.)
        self.assertEqual(build.call_args.args[0], 2.75)
        self.assertEqual(build.call_args.args[1].pair_window_ghz, 60.)
        self.assertEqual(build.call_args.kwargs['distance_um'], 3.35)

    def test_basis_check_can_reverse_tiny_robust_gain(self):
        fields = {}
        for b, fidelity in ((3.1, .99), (3.085, .989), (3.1025, .990001)):
            fields[f'{b:.8f}/40'] = {'nominal': .999,
                'full_grid': {'nominal': .999, 'joint_minimum': fidelity, 'endpoint_objective': 1-fidelity}}
        for b, fidelity in ((3.1, .988), (3.1025, .987)):
            fields[f'{b:.8f}/60'] = {'nominal': .998, 'fixed_same_field_40GHz_nominal': .998,
                'endpoints': {'joint_minimum': fidelity},
                'rows': {k: {'fixed_same_field_40GHz_vertices': np.full((2, 2), fidelity).tolist()}
                    for k in ('0.05/-1', '0.05/1')}}
        result = field.field_comparison({'fields': fields})
        self.assertGreater(result['alternatives'][1]['joint_minimum_change'], 0)
        self.assertLess(result['robust_candidate_joint_change_at_60GHz_fixed_z'], 0)
        self.assertEqual(result['recommendation'], 'retain 3.10 G')


class FinalFigureTests(unittest.TestCase):
    def test_reference_entrypoint_loads_data_records_before_model_build(self):
        """Regression: reference JSON inputs live under data/, not scripts/."""
        with patch.object(field.minimax.shaped, 'build_pair_model',
                side_effect=RuntimeError('model-build-sentinel')) as build:
            with self.assertRaisesRegex(RuntimeError, 'model-build-sentinel'):
                field.minimax._reference_main()
        build.assert_called_once()

    def test_reference_input_contract(self):
        inputs = {'pulse_parameters': field.PARAMETERS.tolist(),
            'numerical_reference_basis': field.asdict(field.p0.REFERENCE_BASIS),
            'field_gauss': 3.1, 'mode_cutoff': 1e-6,
            'propagation_step_ns': .125, 'temperature_k': 0.}
        field.minimax._require_reference_inputs(inputs, field.p0.REFERENCE_BASIS)
        for key, wrong in [('field_gauss', 4.), ('propagation_step_ns', 1.),
                           ('mode_cutoff', 1e-8), ('temperature_k', 300.),
                           ('pulse_parameters', [0]*7),
                           ('numerical_reference_basis', {})]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                field.minimax._require_reference_inputs(inputs | {key: wrong}, field.p0.REFERENCE_BASIS)

    def test_extrema_are_computed_from_samples(self):
        self.assertEqual(field.minimax._curve_summary([2., 3., 4.], [.8, .9, .7]),
            {'minimum': .7, 'minimum_at': 4., 'maximum': .9, 'maximum_at': 3.})


if __name__ == '__main__':
    unittest.main()
