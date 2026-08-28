"""Deterministic checks for the carryover-corrected exact recursions."""

from __future__ import annotations

import random
import unittest
from fractions import Fraction


Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]


def add(*vectors: Vector) -> Vector:
    return tuple(sum(values, Fraction(0)) for values in zip(*vectors))


def subtract(left: Vector, right: Vector) -> Vector:
    return tuple(a - b for a, b in zip(left, right))


def matvec(matrix: Matrix, vector: Vector) -> Vector:
    return tuple(
        sum((coefficient * value for coefficient, value in zip(row, vector)),
            Fraction(0))
        for row in matrix
    )


class CarryoverCorrectedRecursionTests(unittest.TestCase):
    CASES = 200

    def setUp(self) -> None:
        self.rng = random.Random(20260828)

    def scalar(self) -> Fraction:
        return Fraction(self.rng.randint(-15, 15), self.rng.randint(1, 9))

    def vector(self, size: int) -> Vector:
        return tuple(self.scalar() for _ in range(size))

    def matrix(self, size: int) -> Matrix:
        return tuple(self.vector(size) for _ in range(size))

    def test_exact_one_step_recursion_over_200_cases(self) -> None:
        for _ in range(self.CASES):
            size = self.rng.randint(1, 6)
            carryover = self.matrix(size)
            state_a = self.vector(size)
            state_b = self.vector(size)
            update_a = self.vector(size)
            update_b_at_a = self.vector(size)
            update_b_at_b = self.vector(size)

            next_a = add(matvec(carryover, state_a), update_a)
            next_b = add(matvec(carryover, state_b), update_b_at_b)
            forcing = subtract(update_b_at_a, update_a)
            retained = matvec(carryover, subtract(state_b, state_a))
            incremental = subtract(update_b_at_b, update_b_at_a)

            self.assertEqual(
                subtract(next_b, next_a),
                add(forcing, retained, incremental),
            )

    def test_joint_parameter_ema_recursion_over_200_cases(self) -> None:
        for _ in range(self.CASES):
            size = self.rng.randint(1, 6)
            beta = Fraction(self.rng.randint(0, 1000), 1000)
            one_minus_beta = 1 - beta
            theta_a, theta_b = self.vector(size), self.vector(size)
            ema_a, ema_b = self.vector(size), self.vector(size)
            update_a = self.vector(size)
            update_b_at_a = self.vector(size)
            update_b_at_b = self.vector(size)

            theta_next_a = add(theta_a, update_a)
            theta_next_b = add(theta_b, update_b_at_b)
            ema_next_a = add(
                tuple(beta * value for value in ema_a),
                tuple(one_minus_beta * value for value in theta_next_a),
            )
            ema_next_b = add(
                tuple(beta * value for value in ema_b),
                tuple(one_minus_beta * value for value in theta_next_b),
            )

            delta_theta = subtract(theta_b, theta_a)
            delta_ema = subtract(ema_b, ema_a)
            retained_theta = delta_theta
            retained_ema = add(
                tuple(one_minus_beta * value for value in delta_theta),
                tuple(beta * value for value in delta_ema),
            )
            forcing_theta = subtract(update_b_at_a, update_a)
            incremental_theta = subtract(update_b_at_b, update_b_at_a)
            forcing_ema = tuple(
                one_minus_beta * value for value in forcing_theta)
            incremental_ema = tuple(
                one_minus_beta * value for value in incremental_theta)

            self.assertEqual(
                subtract(theta_next_b, theta_next_a),
                add(forcing_theta, retained_theta, incremental_theta),
            )
            self.assertEqual(
                subtract(ema_next_b, ema_next_a),
                add(forcing_ema, retained_ema, incremental_ema),
            )

    def test_finite_horizon_unrolling_over_200_cases(self) -> None:
        for _ in range(self.CASES):
            size = self.rng.randint(1, 5)
            horizon = self.rng.randint(1, 8)
            delta = self.vector(size)
            retained_initial = delta
            propagated_increments: list[Vector] = []

            for _step in range(horizon):
                carryover = self.matrix(size)
                forcing = self.vector(size)
                incremental = self.vector(size)
                new_increment = add(forcing, incremental)

                delta = add(matvec(carryover, delta), new_increment)
                retained_initial = matvec(carryover, retained_initial)
                propagated_increments = [
                    matvec(carryover, value)
                    for value in propagated_increments
                ]
                propagated_increments.append(new_increment)

            self.assertEqual(
                delta,
                add(retained_initial, *propagated_increments),
            )


if __name__ == "__main__":
    unittest.main()
