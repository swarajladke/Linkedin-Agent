"""Planner module implementing the closed decision loop for Pilot.

observe -> diagnose -> generate_actions -> score_actions -> select_actions
-> record_prediction -> execute -> commit
"""
