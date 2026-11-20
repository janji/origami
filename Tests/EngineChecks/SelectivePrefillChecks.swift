import OrigamiEngine
import OrigamiEngineChecks

func selectivePrefillCoveragePolicy() throws { try SelectivePrefillSelfChecks.policy() }
func selectivePrefillRoutingCleanup() throws { try SelectivePrefillSelfChecks.routingAndCleanup() }
func selectivePrefillDemandBypass() throws { try SelectivePrefillSelfChecks.demandBypass() }
func selectivePrefillExactGroups() throws { try SelectivePrefillSelfChecks.exactGroups() }
