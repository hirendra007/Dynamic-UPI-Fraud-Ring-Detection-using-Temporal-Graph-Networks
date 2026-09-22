// ==============================================================================
// 07_fraud_ring_queries.cypher
// Graph Analytics & Fraud Ring Detection Queries for PaySim Transaction Network
// Big Data Analytics Course Project — DA1/DA2 Deliverable
// ==============================================================================

// ------------------------------------------------------------------------------
// QUERY 1: Cyclic Transfer Detection (2 to 4 Hop Closed Cycles / Layering Rings)
// Detects closed loops where money moves through 2, 3, or 4 intermediaries
// and returns to the originating account. This is a classic pattern of money
// laundering (layering) and synthetic identity fraud rings.
// ------------------------------------------------------------------------------

MATCH path = (origin:Account)-[:TRANSFER*2..4]->(origin)
WITH origin, path, [rel in relationships(path) | rel.amount] AS amounts
RETURN 
    origin.id AS ringOrigin,
    length(path) AS cycleLengthHops,
    [node in nodes(path) | node.id] AS ringAccounts,
    reduce(total = 0.0, a in amounts | total + a) AS totalRingVolume,
    path
ORDER BY totalRingVolume DESC
LIMIT 25;

// Alternate 2-Hop Quick Reciprocal Ring Query (Direct ping-pong between 2 accounts):
MATCH (a:Account)-[r1:TRANSFER]->(b:Account)-[r2:TRANSFER]->(a)
WHERE a.id < b.id
RETURN 
    a.id AS account1, 
    b.id AS account2, 
    r1.amount AS amountSent, 
    r2.amount AS amountReturned,
    r1.step AS stepSent,
    r2.step AS stepReturned
LIMIT 20;


// ------------------------------------------------------------------------------
// QUERY 2: High-Degree "Hub" & "Mule" Accounts (Fan-In / Fan-Out Topologies)
// Identifies accounts with an unusually disproportionate ratio of incoming vs
// outgoing transfers:
// - Fan-In Mules: Many distinct accounts sending funds into a single accumulator.
// - Fan-Out Dispersers: A single account dispersing funds into dozens of beneficiaries.
// ------------------------------------------------------------------------------

MATCH (a:Account)
WITH a, 
     count{(a)-[:TRANSFER]->()} AS outDegree,
     count{(a)<-[:TRANSFER]-()} AS inDegree,
     count{(a)-[:TRANSFER {isFraud: 1}]->()} + count{(a)<-[:TRANSFER {isFraud: 1}]-()} AS fraudIncidentDegree
WHERE outDegree > 5 OR inDegree > 5 OR fraudIncidentDegree > 0
RETURN 
    a.id AS accountId,
    labels(a) AS accountLabels,
    inDegree AS totalInflowTransactions,
    outDegree AS totalOutflowTransactions,
    inDegree + outDegree AS totalDegree,
    fraudIncidentDegree,
    CASE 
        WHEN inDegree > 10 AND outDegree <= 2 THEN "Smurfing / Mule Accumulator (Fan-In)"
        WHEN outDegree > 10 AND inDegree <= 2 THEN "Disperser / Cash-Out Mule (Fan-Out)"
        WHEN inDegree > 5 AND outDegree > 5 THEN "High-Velocity Transit Hub"
        ELSE "Active Fraud Participant"
    END AS muleProfile
ORDER BY totalDegree DESC
LIMIT 25;


// ------------------------------------------------------------------------------
// QUERY 3: Smurfing / Structuring Detection (Temporal Inflow Aggregation)
// Detects money laundering smurfing where multiple senders make smaller transfers
// to a central mule within a narrow time window, followed by a large exit transfer.
// ------------------------------------------------------------------------------

MATCH (mule:Account)<-[in_tx:TRANSFER]-(sender:Account)
WHERE mule.id <> sender.id
WITH mule, 
     count(DISTINCT sender) AS uniqueSenders, 
     collect(in_tx.amount) AS incomingAmounts,
     min(in_tx.step) AS firstDepositStep,
     max(in_tx.step) AS lastDepositStep
WHERE uniqueSenders >= 3 AND (lastDepositStep - firstDepositStep) <= 24
OPTIONAL MATCH (mule)-[out_tx:TRANSFER]->(destination:Account)
RETURN 
    mule.id AS intermediaryMule,
    uniqueSenders,
    reduce(s = 0.0, x in incomingAmounts | s + x) AS totalInflowDeposited,
    count(out_tx) AS exitTransfersCount,
    collect(out_tx.amount) AS exitTransferAmounts
ORDER BY uniqueSenders DESC
LIMIT 20;


// ------------------------------------------------------------------------------
// QUERY 4: Community Detection / Weakly Connected Components (GDS or Cypher)
// Groups collusive fraud accounts into densely connected clusters.
// ------------------------------------------------------------------------------

// If Neo4j Graph Data Science (GDS) library is installed:
// 1. Project sub-graph:
// CALL gds.graph.project(
//     'fraudRingNetwork',
//     'Account',
//     'TRANSFER',
//     { relationshipProperties: ['amount'] }
// );
// 
// 2. Run Louvain Community Detection:
// CALL gds.louvain.stream('fraudRingNetwork')
// YIELD nodeId, communityId
// RETURN gds.util.asNode(nodeId).id AS accountId, communityId
// ORDER BY communityId ASC;

// Pure Cypher Connected Sub-Graph Query (Works without GDS installed):
MATCH (flagged:Account:Flagged)
MATCH path = (flagged)-[:TRANSFER*1..3]-(connected:Account)
WITH flagged, collect(DISTINCT connected.id) AS clusterMembers, count(DISTINCT connected) AS clusterSize
WHERE clusterSize >= 2
RETURN 
    flagged.id AS seedFraudAccount,
    clusterSize,
    clusterMembers[0..10] AS sampleClusterMembers
ORDER BY clusterSize DESC
LIMIT 15;


// ------------------------------------------------------------------------------
// QUERY 5: Complete Ring Visualizer for Neo4j Browser Export
// Run this directly in the Neo4j Browser (http://localhost:7474) to generate
// a visual graph showing fraud rings connected to Flagged accounts.
// ------------------------------------------------------------------------------

MATCH path = (f1:Account:Flagged)-[r:TRANSFER]->(f2:Account)
OPTIONAL MATCH secondHop = (f2)-[:TRANSFER]->(f3:Account)
RETURN path, secondHop
LIMIT 50;
