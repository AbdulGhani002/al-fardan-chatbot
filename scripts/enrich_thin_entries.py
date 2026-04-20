"""Enrich the most heavily-routed im-* entries with canonical facts.

The 450 im-* entries ingested in commit d2e7b6e include terse placeholder
answers on high-stakes topics (Sharia, VARA, Proof of Reserves, custody,
insurance). On the VPS these generic answers outrank the curated af-* /
jq-* / if-* entries because dense retrieval matches on question similarity
without knowing which answer has more substance.

This script rewrites the answers only — aliases and keywords stay put so
retrieval still finds them — replacing each terse placeholder with a
fact-dense institutional reply drawing on the canonical facts in
24_authoritative_facts.jsonl.

Run once; commit; reindex on the VPS.
"""

from __future__ import annotations

import json
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

KB = Path(r"C:\CC\Code\al-fardan-chatbot\app\data\kb")


# Canonical replacement answers — institutional register, fact-dense.
# Map: entry id → new answer text.
REPLACEMENTS: dict[str, str] = {
    # ─── Company identity ─────────────────────────────────────────────
    "im-004": (
        "Al-Fardan Q9 is the institutional digital-asset platform of the Al-Fardan "
        "Group. We provide custody, staking, OTC execution, and asset-backed "
        "lending to family offices, HNWIs, asset managers, and hedge funds. "
        "Operations run under a VARA-licensed framework via our infrastructure "
        "partner Fuze (VL/23/10/002) and are DIFC-registered (#5605). The parent "
        "Al-Fardan Exchange has been CBUAE-regulated in the UAE since 1971."
    ),
    "im-005": (
        "Al-Fardan Q9 operates within a documented regulatory framework: VARA-"
        "licensed digital-asset activity through our infrastructure partner "
        "Fuze (VL/23/10/002, issued 17 November 2023), DIFC-registered (#5605), "
        "and the parent Al-Fardan Exchange has been CBUAE-regulated since 1971. "
        "Compliance certifications — SOC 2 Type II, ISO 27001, ISO 27017, and "
        "ISO 9001 — are re-audited annually."
    ),
    "im-006": (
        "VARA Licence VL/23/10/002 is held by Fuze (Morpheus Software Technology "
        "FZE), our infrastructure partner. Issued 17 November 2023 by the Virtual "
        "Assets Regulatory Authority in Dubai. Publicly verifiable on the VARA "
        "register at vara.ae."
    ),
    "im-007": (
        "Yes. Al-Fardan Q9 is registered at the Dubai International Financial "
        "Centre under registration number 5605. Publicly verifiable at "
        "difc.ae/public-register."
    ),
    "im-008": (
        "Al-Fardan Q9 is headquartered at The Onyx Tower 2, The Greens, Sheikh "
        "Zayed Road, Dubai, United Arab Emirates. In-person meetings are "
        "available by prior appointment through the relationship manager."
    ),
    "im-009": (
        "Four institutional services: Custody (Fireblocks MPC infrastructure, "
        "segregated vaults, Lloyd's-insured up to USD 250M), Staking (ETH, SOL, "
        "and other major proof-of-stake assets with daily reward crediting), "
        "OTC Desk (block settlement with bilateral execution and tight spreads), "
        "and Lending (Shariah-compliant Murabaha credit lines from 3.25% APR, "
        "BTC/ETH collateral, up to 75% LTV)."
    ),
    "im-011": (
        "Al-Fardan Q9 serves family offices, HNWIs, asset managers, and hedge "
        "funds across the GCC and globally. Specific client counts are treated "
        "as confidential and can be discussed with the relationship manager "
        "under NDA."
    ),

    # ─── Custody / security ───────────────────────────────────────────
    "im-019": (
        "Yes. Institutional-grade custody through Fireblocks MPC infrastructure "
        "— segregated vaults, no rehypothecation, and legal separation from our "
        "balance sheet. Coverage up to USD 250M via the Lloyd's of London policy "
        "(SY-2025-49881), subject to policy terms and exclusions."
    ),
    "im-020": (
        "Yes. Custodied assets are insured up to USD 250M under a Lloyd's of "
        "London policy (SY-2025-49881), covering theft, hacks, insider "
        "collusion, and loss of private keys, subject to policy terms and "
        "exclusions. Market price movements remain the client's own risk."
    ),
    "im-021": (
        "Assets are secured via Fireblocks MPC (multi-party computation) "
        "custody — key shards distributed across multiple parties, so no single "
        "party can move funds unilaterally. Supplemented by a USD 250M Lloyd's "
        "policy and segregation from our balance sheet under a bankruptcy-remote "
        "structure."
    ),

    # ─── Sharia (the one James saw fail) ──────────────────────────────
    "im-026": (
        "Yes. All products are reviewed by our independent Sharia Supervisory "
        "Board, chaired by Sheikh Dr. Tariq Al-Mahrouqi — a member of the AAOIFI "
        "(Accounting and Auditing Organization for Islamic Financial "
        "Institutions) Sharia Board. Lending is structured as Murabaha (disclosed "
        "cost, not interest/riba). Staking rewards are compensation for network "
        "security services, not interest. Full Sharia reports are available on "
        "our website under Shariah Reports."
    ),

    # ─── Certifications ───────────────────────────────────────────────
    "im-027": (
        "Al-Fardan Q9 holds SOC 2 Type II (security, availability, and "
        "confidentiality), ISO 27001 (Information Security Management), ISO "
        "27017 (Cloud Security), and ISO 9001 (Quality Management). All four "
        "are re-audited annually by independent assessors. Certificates are "
        "available to qualified institutional clients on request under NDA."
    ),

    # ─── Infrastructure / partners ────────────────────────────────────
    "im-032": (
        "Fireblocks is the institutional digital-asset custody and settlement "
        "platform used for our custody infrastructure. It uses MPC (multi-party "
        "computation) to distribute key-signing authority, eliminating single "
        "points of failure. Fireblocks is widely adopted by banks, exchanges, "
        "and asset managers globally."
    ),
    "im-033": (
        "MPC (multi-party computation) custody distributes the signing "
        "authority across multiple independent parties. No single party ever "
        "holds a complete private key, so no single compromise can move funds. "
        "This is architecturally different from hot-wallet exchange custody "
        "and from seed-phrase self-custody."
    ),
    "im-034": (
        "Safety rests on four layers: (1) Fireblocks MPC custody — key shards "
        "distributed, no single point of failure; (2) legal segregation — "
        "assets are client-owned, not on our balance sheet; (3) USD 250M "
        "Lloyd's insurance policy (SY-2025-49881) covering theft, hacks, "
        "insider collusion, and key loss; (4) regulated framework — VARA "
        "(via Fuze), DIFC, SOC 2, ISO 27001/27017/9001."
    ),

    # ─── Proof of reserves ────────────────────────────────────────────
    "im-035": (
        "Yes. Reserve coverage is maintained at 215% across approximately USD "
        "2.55B in segregated Fireblocks MPC vaults, audited quarterly by an "
        "independent firm. Truncated wallet addresses are visible in the client "
        "dashboard. Full wallet addresses and auditor letters are available to "
        "qualified institutional clients under NDA."
    ),

    # ─── Fuze partnership ─────────────────────────────────────────────
    "im-039": (
        "Fuze (Morpheus Software Technology FZE) is our VARA-licensed "
        "infrastructure partner. Fuze holds VARA Licence VL/23/10/002 (issued "
        "17 November 2023). The regulated digital-asset activity — custody, "
        "execution, buying, selling, holding, and transfer — runs through "
        "Fuze's licensed stack. An MoU was signed 10 December 2023 between "
        "Al-Fardan Exchange and Fuze."
    ),
    "im-052": (
        "Three public sources and one private: VARA Licence via "
        "vara.ae/en/licenses-and-register/public-register (search 'Morpheus "
        "Software Technology FZE' or 'Fuze'); DIFC Registration via "
        "difc.ae/public-register (search '5605' or 'Al Fardan'); MoU announcement "
        "in Gulf News (December 2023). SOC 2 and ISO certificates are provided "
        "to qualified institutional clients under NDA."
    ),
    "im-053": (
        "Yes. SOC 2 Type II (security, availability, confidentiality), ISO "
        "27001 (Information Security Management), ISO 27017 (Cloud Security), "
        "and ISO 9001 (Quality Management) — re-audited annually by independent "
        "assessors. Audit letters can be shared with qualified institutional "
        "clients under NDA."
    ),

    # ─── Cold storage ─────────────────────────────────────────────────
    "im-065": (
        "Yes. Approximately 95% of custodied assets are held in air-gapped cold "
        "storage via Fireblocks MPC architecture, with only operational balances "
        "in warm segments. This is the same model used by major institutional "
        "custodians globally."
    ),

    # ─── Insurance ────────────────────────────────────────────────────
    "im-107": (
        "Yes — custodied assets are insured under a USD 250M Lloyd's of London "
        "policy (SY-2025-49881) covering theft, hacks, insider collusion, and "
        "loss of private keys. Market price movements and counterparty/credit "
        "risk are NOT covered. Full policy terms and exclusions are available "
        "to qualified institutional clients under NDA."
    ),
    "im-108": (
        "Yes. Audit reports (SOC 2 Type II, ISO 27001/27017/9001, and quarterly "
        "proof-of-reserves) are made available to qualified institutional "
        "clients under NDA through the relationship manager."
    ),

    # ─── MPC custody details ─────────────────────────────────────────
    "im-176": (
        "We use Fireblocks MPC (multi-party computation) custody. Private-key "
        "signing authority is cryptographically distributed across multiple "
        "parties with no single point of compromise. ~95% of assets sit in "
        "air-gapped cold segments; operational hot balances are kept minimal. "
        "All vaults are bankruptcy-remote and legally segregated from our "
        "balance sheet."
    ),
    "im-177": (
        "Private keys are never assembled in a single location. MPC splits the "
        "signing operation across multiple independent parties; each holds a "
        "key share but no party ever sees or reconstructs the full key. This "
        "is architecturally resistant to single-point compromise."
    ),
    "im-178": (
        "Yes. Approximately 95% of custodied assets are held in air-gapped cold "
        "segments within the Fireblocks MPC architecture. Only minimal "
        "operational balances are kept in warm segments for settlement liquidity."
    ),
    "im-180": (
        "Yes. MPC custody inherently provides multi-party signing — each "
        "transaction requires a quorum of independent key shares. This is a "
        "stronger security model than traditional multi-sig wallets because no "
        "full private key ever exists at any point."
    ),
    "im-181": (
        "Custodied assets are covered under the USD 250M Lloyd's of London "
        "policy (SY-2025-49881), subject to stated terms and exclusions. The "
        "policy covers theft, hacks, insider collusion, and loss of private "
        "keys, but not market price movements or counterparty/credit risk."
    ),
    "im-182": (
        "Our defence-in-depth approach — Fireblocks MPC with no single point of "
        "failure, 95% air-gapped cold storage, legal segregation, and a USD "
        "250M Lloyd's policy covering theft/hacks/insider collusion — is "
        "designed to make breaches unlikely and recoverable. Incident response "
        "protocols are audited under SOC 2. Specific claim mechanics escalate "
        "to the claims and operations teams."
    ),

    # ─── Partners / Fuze ─────────────────────────────────────────────
    "im-200": (
        "Yes. Our infrastructure stack uses regulated, audited partners. Fuze "
        "(VARA-licensed, VL/23/10/002) provides the regulated digital-asset "
        "infrastructure. Fireblocks provides the MPC custody layer. Lloyd's of "
        "London underwrites the custody insurance policy (SY-2025-49881)."
    ),
    "im-201": (
        "Key partners: Fuze (VARA-licensed infrastructure partner, VL/23/10/002) "
        "for regulated digital-asset activity; Fireblocks for MPC custody; "
        "Lloyd's of London for the USD 250M custody insurance policy "
        "(SY-2025-49881); independent assessors for SOC 2 and ISO audits."
    ),
    "im-202": (
        "Regulated digital-asset activity runs through our infrastructure "
        "partner Fuze (Morpheus Software Technology FZE), which holds VARA "
        "Licence VL/23/10/002. An MoU formalising the collaboration was signed "
        "10 December 2023 between Al-Fardan Exchange L.L.C. and Fuze."
    ),
    "im-204": (
        "Yes. Fuze (Morpheus Software Technology FZE) holds VARA Licence "
        "VL/23/10/002, issued 17 November 2023 by the Virtual Assets Regulatory "
        "Authority in Dubai. Publicly verifiable on the VARA register at vara.ae."
    ),

    # ─── Audit trails ────────────────────────────────────────────────
    "im-278": (
        "Yes. Independent auditors are supported throughout their engagements — "
        "we provide access to transaction records, reserve reports, and "
        "infrastructure documentation under NDA. Coordination runs through the "
        "relationship manager and compliance team."
    ),
    "im-279": (
        "Yes. Full audit trails are maintained across every custody movement, "
        "trade, staking reward, and lending event. Trails are available to "
        "qualified institutional clients under NDA, and are used for SOC 2 and "
        "ISO audit evidence annually."
    ),
}


def main() -> None:
    touched_files = set()
    replaced = 0
    missed: list[str] = []

    for path in sorted(KB.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            lines = f.readlines()

        changed = False
        out_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                out_lines.append(line)
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                out_lines.append(line)
                continue
            eid = entry.get("id", "")
            if eid in REPLACEMENTS:
                entry["answer"] = REPLACEMENTS[eid]
                out_lines.append(json.dumps(entry, ensure_ascii=False) + "\n")
                replaced += 1
                changed = True
            else:
                out_lines.append(line)

        if changed:
            touched_files.add(path.name)
            with path.open("w", encoding="utf-8") as f:
                f.writelines(out_lines)

    for eid in REPLACEMENTS:
        # We can't easily know if we missed any, so report post-facto
        pass

    print(f"Enriched {replaced}/{len(REPLACEMENTS)} entries")
    print(f"Touched files: {sorted(touched_files)}")


if __name__ == "__main__":
    main()
