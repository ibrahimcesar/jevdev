// jevdev default execution policy.
//
// Cedar semantics: any matching forbid denies. A matching permit allows.
// When neither matches, the harness asks (Jev's opinion first, then a human).
//
// Resource attributes are computed by the harness before evaluation:
//   command, program, tool, access, touches (set of paths), touches_sensitive,
//   outside_root, read_only, is_test, destructive, script_egress
// Context: task_scope, goal

@id("deny_sensitive_paths")
forbid (principal, action, resource)
when { resource.touches_sensitive };

@id("deny_destructive")
forbid (principal, action == Action::"exec", resource)
when { resource.destructive };

@id("deny_script_egress_outside_deploy")
forbid (principal, action == Action::"exec", resource)
when { resource.script_egress && context.task_scope != "deploy" };

@id("allow_read_only_programs")
permit (principal, action == Action::"exec", resource)
when { resource.read_only && !resource.outside_root };

@id("allow_tests")
permit (principal, action == Action::"exec", resource)
when { resource.is_test && !resource.outside_root };

@id("allow_reads_in_repo")
permit (principal, action == Action::"read", resource)
when { !resource.outside_root };

@id("allow_writes_in_repo")
permit (principal, action == Action::"write", resource)
when { !resource.outside_root };
