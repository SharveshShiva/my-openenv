from pydantic import BaseModel, Field
from typing import Literal, List, Optional

class ClaimData(BaseModel):
    claim_id: str
    patient_name: str
    age: int
    treatment: str
    hospital: str
    claim_amount: float
    policy_duration_months: int
    documents_submitted: List[str]
    history_claims_count: int

class Observation(BaseModel):
    claim: ClaimData
    difficulty: str

class Action(BaseModel):
    decision: Literal["APPROVE", "REJECT", "ESCALATE"] = Field(..., description="The final decision on the claim")
    fraud_score: float = Field(..., ge=0.0, le=1.0, description="The estimated likelihood of fraud between 0.0 and 1.0")
    reasoning: str = Field(..., description="The reasoning behind the decision and fraud score")

class Reward(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    details: Optional[dict] = None
