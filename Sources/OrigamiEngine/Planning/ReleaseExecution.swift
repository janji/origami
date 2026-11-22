// The execution switches choose the kernels, so this file is in the arithmetic digest (`ArithmeticDigest`); the rest of
// the plan only chooses sizes.
extension OrigamiPlan {
    /// The v1 release execution switches (8K); 32K adds the fused compact prefill attention.
    static let releaseExecution = #"""
        {"batched_pinned_attention":true,"batched_rope":true,"batched_selector":true,"decode_early_demand":true,"decode_preencoded_layers":true,
         "decode_next_layer_prerouting":true,"decode_ple_internal_race":true,"decode_prerouting_margin":0.1,"decode_prerouting_override_width":10,
         "decode_prerouting_read_hygiene":true,"decode_prerouting_slot_override":true,"decode_token_plan":true,"dense_resident_bytes":382205952,
         "device_ple":true,"elastic_decode_residency":true,"expert_command_batch":8,"expert_decode_source_policy":"earliestCompletion",
         "expert_layer_waves":8,"expert_mode":"slabPackedPreferred","expert_prefill_layer_command":true,"expert_prefill_read_ahead":true,
         "expert_prefill_source_policy":"earliestCompletion","expert_reuse_admission":true,"expert_scratch_reuse":true,"expert_slots_per_slab":8,
         "expert_source_online_bounds":{"ceiling_multiple":4.0,"delay_ceiling_multiple":1.0,"floor_multiple":0.25,"smoothing":0.02,"warmup_observations":32},
         "expert_source_online_calibration":true,"expert_tail_replicas":true,"expert_window":16,"fused_experts":true,"fused_gdn_min_rows":8,
         "fused_gdn_preparation":true,"fused_hc_decode":true,"fused_hc_epilogue":true,"fused_hc_min_rows":8,"fused_hc_prefill":true,
         "fused_residual_injection":true,"fused_residual_min_rows":8,"fused_selected_prefill_attention":false,"gathered_expert_decode":true,
         "gdn_buffer_region":true,"gdn_device_block":true,"gdn_exact_kernels":true,"gdn_input_composition":true,"gdn_resident_state":true,
         "gpu_keepalive":true,"grouped_expert_bm32":true,"grouped_expert_bm32_min_rows":32,"grouped_expert_mma_min_rows":192,
         "grouped_expert_prefill":true,"grouped_expert_tiled":true,"hc_decode_fusion":true,"hc_exact_fusion":true,"host_stage_prefetch":true,
         "io_queue_depth":4,"io_reader_pool":true,"moe_device_prefill":true,"moe_shared_read_overlap":true,"pinned_attention":true,
         "pinned_dense":true,"pinned_indexer":true,"pinned_reductions":true,"pinned_rms":true,"pinned_rope":true,"pinned_unary":true,
         "preallocate_decode_slabs":true,"prefill_double_buffer":true,"prefill_ple_overlap":true,"qsa_decode_reuse":true,"qsa_exact_kernels":true,
         "quantized_bm32_min_rows":256,"quantized_mma_bf16":true,"quantized_tiled":true,"quantized_tiled_bm32":true,"quantized_tiled_prefetch":true,
         "ragged_prefill_scope":true,"register_qsa_decode":true,"resident_small_tensors":true,"resident_tensor_bytes":2580086784,
         "simd_matrix":true,"whole_execution_schedule":true,"wire_decode_slabs":true}
        """#
}
