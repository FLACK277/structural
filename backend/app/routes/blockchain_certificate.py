from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
import os
from dotenv import load_dotenv
from thirdweb import ThirdwebSDK
from thirdweb.types import NFTMetadataInput

load_dotenv()

router = APIRouter()

class CertificateRequest(BaseModel):
    name: str
    skill: str

@router.post("/issue_certificate")
def issue_certificate(request: CertificateRequest):
    try:
        private_key = os.getenv("PRIVATE_KEY")
        contract_address = os.getenv("CONTRACT_ADDRESS")
        if not private_key or not contract_address:
            raise Exception("Missing PRIVATE_KEY or CONTRACT_ADDRESS in environment.")

        # Connect to Amoy testnet
        sdk = ThirdwebSDK("amoy")
        sdk.wallet.connect(private_key)
        contract = sdk.get_nft_collection(contract_address)

        issued_at = datetime.utcnow().isoformat() + 'Z'
        metadata = NFTMetadataInput(
            name=f"Skill Certificate: {request.skill}",
            description=f"SkillBridge certificate for {request.name} in {request.skill}",
            properties={
                "name": request.name,
                "skill": request.skill,
                "issued_at": issued_at
            }
        )

        tx = contract.mint(metadata)
        token_id = tx["id"]
        tx_hash = tx["receipt"]["transactionHash"]

        return {
            "name": request.name,
            "skill": request.skill,
            "certificate_id": str(token_id),
            "blockchain_proof": tx_hash,
            "issued_at": issued_at,
            "contract_address": contract_address,
            "explorer_url": f"https://www.oklink.com/amoy/tx/{tx_hash}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error issuing certificate: {e}")