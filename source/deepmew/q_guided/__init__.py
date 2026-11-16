"""Research-only Q-guided teacher primitives."""

from deepmew.q_guided.branching import (
    PRIVATE_CONTENT_SENTINEL,
    LegalSelectionCapExceeded,
    PrivilegedTeacherExecutionError,
    PrivilegedTeacherReceipt,
    RootConsistencyReceipt,
    assert_private_content_absent,
    build_root_consistency_receipt,
    count_legal_selections,
    enumerate_legal_selections,
    execute_privileged_teacher,
    find_private_content,
    project_student_input,
    sample_uniform_legal_selection,
    select_epsilon_mixture,
    student_input_bytes,
)

__all__ = [
    "PRIVATE_CONTENT_SENTINEL",
    "LegalSelectionCapExceeded",
    "PrivilegedTeacherExecutionError",
    "PrivilegedTeacherReceipt",
    "RootConsistencyReceipt",
    "assert_private_content_absent",
    "build_root_consistency_receipt",
    "count_legal_selections",
    "enumerate_legal_selections",
    "execute_privileged_teacher",
    "find_private_content",
    "project_student_input",
    "sample_uniform_legal_selection",
    "select_epsilon_mixture",
    "student_input_bytes",
]
