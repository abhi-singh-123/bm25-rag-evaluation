"""
Retrieval Evaluation: BM25 vs Vector Search
============================================
Evaluates both retrieval approaches across two independent corpora:

  Corpus 1: Prometheus Operator runbooks (Apache 2.0)
            https://github.com/prometheus-operator/runbooks

  Corpus 2: Kubernetes official troubleshooting documentation (Apache 2.0)
            https://kubernetes.io/docs/tasks/debug/

Usage:
    pip install -r requirements.txt
    python eval_unified.py

Outputs:
    eval_unified_results.json   all raw numbers
    eval_unified_results.md     formatted tables ready to paste into article
"""

import json, time
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

VECTOR_MODEL = "all-MiniLM-L6-v2"

# ─────────────────────────────────────────────────────────────────────────────
# CORPUS 1 — Prometheus Operator runbooks
# Source: https://github.com/prometheus-operator/runbooks (Apache 2.0)
# ─────────────────────────────────────────────────────────────────────────────
CORPUS_1 = [
    # 0
    ("KubePodCrashLooping",
     "Alert: KubePodCrashLooping. A pod is crash-looping, meaning the container "
     "starts, crashes, and restarts repeatedly. Triage steps: run kubectl describe pod "
     "to check Events and Last State exit codes. Run kubectl logs --previous to view "
     "the last crash output. Common causes include misconfigured environment variables, "
     "missing secrets or configmaps, OOMKilled status due to memory limits, and "
     "application startup errors. If the exit code is 137, the process was killed by "
     "the kernel, check memory limits. If the exit code is 1, inspect application logs. "
     "Increase resource limits if OOMKilled. Fix the underlying application error and "
     "redeploy if exit code is non-zero from the app itself."),
    # 1
    ("KubePodNotReady",
     "Alert: KubePodNotReady. One or more pods have been in a non-ready state for "
     "longer than 15 minutes. Triage: run kubectl get pods to identify affected pods. "
     "Run kubectl describe pod to check conditions and events. Check if readiness probes "
     "are failing, verify the probe endpoint is reachable and returning 200. Check if "
     "dependent services like databases or configmaps are available. If the pod is "
     "pending, check for resource quota exhaustion or node affinity issues. "
     "If the pod is stuck in ContainerCreating, check for image pull errors."),
    # 2
    ("KubeNodeNotReady",
     "Alert: KubeNodeNotReady. A Kubernetes node has been in the NotReady state for "
     "more than 15 minutes. Triage: run kubectl describe node to check conditions. "
     "SSH into the node and check kubelet status with systemctl status kubelet. "
     "Check kubelet logs with journalctl -u kubelet. Common causes: disk pressure, "
     "memory pressure, network partition, or kubelet crash. If disk pressure, free up "
     "space by pruning unused images with crictl rmi. If kubelet is down, restart it. "
     "If the node is unreachable, check cloud provider console for instance health."),
    # 3
    ("KubeNodeUnreachable",
     "Alert: KubeNodeUnreachable. A node is unreachable and workloads may be evicted. "
     "Check cloud provider console for instance status immediately. Verify network "
     "connectivity from other nodes. Check VPC security groups and NACLs. If the node "
     "is truly down, drain and cordon it to prevent scheduling. Investigate if the "
     "outage is isolated or part of a broader availability zone issue. "
     "Do not forcibly delete pods until you confirm the node is truly dead, "
     "to avoid split-brain with stateful workloads."),
    # 4
    ("KubeDeploymentReplicasMismatch",
     "Alert: KubeDeploymentReplicasMismatch. A deployment has fewer ready replicas "
     "than desired for more than 15 minutes. Check kubectl get deployment and "
     "kubectl describe deployment for rollout status. Look for failed pods with "
     "kubectl get pods. Check recent rollout history with kubectl rollout history. "
     "If a recent deploy caused the issue, roll back with kubectl rollout undo. "
     "Check resource quotas, the namespace may be at capacity. "
     "Check for PodDisruptionBudget violations blocking scale-up."),
    # 5
    ("KubeStatefulSetReplicasMismatch",
     "Alert: KubeStatefulSetReplicasMismatch. A StatefulSet does not have the expected "
     "number of ready replicas. StatefulSets scale sequentially, check if a previous "
     "pod is stuck before the next one can start. Run kubectl get pods to find the "
     "stuck pod. Check PersistentVolumeClaims, if a PVC is pending, the pod cannot "
     "start. Verify storage class is available and the cloud provisioner is healthy. "
     "Check pod logs for storage mount errors."),
    # 6
    ("KubeJobFailed",
     "Alert: KubeJobFailed. A Kubernetes Job has failed. Run kubectl describe job to "
     "see failure reason. Check pod logs from the failed job pods. Common causes: "
     "application error, exceeded backoffLimit, or resource exhaustion. "
     "If the job is a CronJob, check the CronJob schedule and last successful run time. "
     "Check if the job image is still accessible. If the failure is transient, "
     "delete the failed job and let the CronJob controller create a new one. "
     "If persistent, fix the underlying issue before retrying."),
    # 7
    ("KubeCPUOvercommit",
     "Alert: KubeCPUOvercommit. The cluster has overcommitted CPU resource requests "
     "and cannot tolerate a node failure. This means total CPU requests across all pods "
     "exceed total allocatable CPU across all nodes minus one. Review namespaces with "
     "highest CPU requests using kubectl top nodes and kubectl top pods. "
     "Consider reducing CPU requests on non-critical workloads. "
     "Add nodes to the cluster if requests reflect real usage. "
     "Set LimitRanges to prevent runaway CPU requests in new workloads."),
    # 8
    ("KubeMemoryOvercommit",
     "Alert: KubeMemoryOvercommit. The cluster has overcommitted memory requests and "
     "cannot survive a node failure. Total memory requests exceed total allocatable "
     "memory across all nodes minus one. This increases OOMKill risk during node loss. "
     "Identify pods with high memory requests using kubectl top pods --all-namespaces. "
     "Reduce memory requests on workloads where actual usage is well below the request. "
     "Add nodes or upgrade instance types if requests are accurate reflections of need."),
    # 9
    ("KubeContainerWaiting",
     "Alert: KubeContainerWaiting. A container has been in a waiting state for over "
     "one hour. Run kubectl describe pod to check the waiting reason. "
     "Common reasons: ImagePullBackOff means the image cannot be pulled, check image "
     "name, tag, and registry credentials. CrashLoopBackOff means the container keeps "
     "failing on startup, check logs. CreateContainerConfigError means a secret or "
     "configmap referenced by the pod does not exist, verify references in pod spec."),
    # 10
    ("KubePersistentVolumeFillingUp",
     "Alert: KubePersistentVolumeFillingUp. A PersistentVolume is expected to fill up "
     "within four days based on current usage trend. Identify which pod is using the PV "
     "with kubectl get pvc --all-namespaces. Investigate what is filling the volume, "
     "logs, database data, or temporary files. For log volumes, implement log rotation. "
     "For database volumes, consider archiving old data. Expand the PVC if the storage "
     "class supports volume expansion: edit the PVC spec and increase the storage size. "
     "Verify the underlying cloud disk was resized with kubectl describe pvc."),
    # 11
    ("KubePersistentVolumeErrors",
     "Alert: KubePersistentVolumeErrors. One or more PersistentVolumes are in a failed "
     "or pending state. Run kubectl get pv and kubectl describe pv to see error details. "
     "Common causes: cloud disk was deleted or detached outside Kubernetes, "
     "storage class provisioner is unhealthy, or IAM permissions for the cloud provider "
     "CSI driver have changed. Check the CSI driver pods in kube-system. "
     "For detached volumes, re-attach at the cloud provider level then trigger "
     "reconciliation by deleting and recreating the PVC if needed."),
    # 12
    ("KubeAPIErrorsHigh",
     "Alert: KubeAPIErrorsHigh. The Kubernetes API server is returning a high rate of "
     "errors. Check API server logs with kubectl logs -n kube-system. "
     "Look for 5xx responses in metrics using the apiserver_request_total counter "
     "filtered by code=~5xx. Common causes: etcd latency or unavailability, "
     "API server overload from too many watchers, or a misbehaving controller. "
     "Check etcd health with etcdctl endpoint health. If etcd is healthy, "
     "identify the highest-rate clients using audit logs."),
    # 13
    ("KubeAPILatencyHigh",
     "Alert: KubeAPILatencyHigh. API server request latency is elevated. "
     "High latency can cause cascading failures as controllers time out waiting for "
     "API responses. Check if etcd is slow, etcd latency directly impacts API latency. "
     "Run etcdctl check perf. Check for expensive list operations flooding the API "
     "server, look for requests with large response sizes in audit logs. "
     "Consider enabling API Priority and Fairness to rate-limit expensive requests. "
     "Vertical scaling of the API server may be needed under heavy load."),
    # 14
    ("KubeClientErrors",
     "Alert: KubeClientErrors. Clients are encountering a high rate of errors when "
     "communicating with the API server. Identify which clients are failing by checking "
     "the user_agent label on the apiserver_request_total metric. Common culprits: "
     "controllers with stale informer caches, helm operations, or CI/CD pipelines. "
     "Check if the errors are authentication 401, authorization 403, or server "
     "errors 5xx. For 401/403, check RBAC configuration and service account tokens. "
     "For 5xx, focus on API server and etcd health."),
    # 15
    ("PrometheusNotConnectedToAlertmanager",
     "Alert: PrometheusNotConnectedToAlertmanager. Prometheus is not connected to any "
     "Alertmanager. Alerts will not be delivered. Check the Alertmanager service is "
     "running: kubectl get pods -n monitoring. Verify the Prometheus alerting config "
     "points to the correct Alertmanager address. Check for network policies blocking "
     "traffic between Prometheus and Alertmanager. Check Prometheus logs for "
     "connection refused or timeout errors. Verify the Alertmanager service DNS is "
     "resolving correctly from the Prometheus pod."),
    # 16
    ("PrometheusRuleFailures",
     "Alert: PrometheusRuleFailures. Prometheus is failing to evaluate some alerting "
     "or recording rules. Run kubectl describe prometheusrule to find malformed rules. "
     "Check Prometheus logs for rule evaluation errors. Common causes: invalid PromQL "
     "syntax, references to non-existent metrics, or label matchers that never match. "
     "Fix the rule definition and the PrometheusRule resource will be hot-reloaded. "
     "Verify the fix by checking the /rules endpoint in the Prometheus UI."),
    # 17
    ("AlertmanagerFailedReload",
     "Alert: AlertmanagerFailedReload. Alertmanager failed to reload its configuration. "
     "The previous configuration is still active. Check Alertmanager logs for the "
     "specific parse error. Run kubectl describe secret alertmanager-config to verify "
     "the secret exists. Validate the alertmanager config locally using "
     "amtool check-config. Common errors: invalid YAML, missing receiver reference, "
     "or invalid route matcher syntax. Fix the config in the secret and "
     "Alertmanager will automatically attempt another reload."),
    # 18
    ("TargetDown",
     "Alert: TargetDown. One or more Prometheus scrape targets are down. "
     "Identify which targets are down in the Prometheus UI under Status > Targets. "
     "Check if the pod or service the target belongs to is running. "
     "Verify the ServiceMonitor or PodMonitor selector matches the target labels. "
     "Check for network policies blocking Prometheus scrape traffic. "
     "Verify the metrics port and path are correct in the ServiceMonitor spec. "
     "Check if the application's /metrics endpoint is healthy by curling it directly."),
    # 19
    ("NodeFilesystemAlmostOutOfSpace",
     "Alert: NodeFilesystemAlmostOutOfSpace. A node filesystem is predicted to run out "
     "of space within 24 hours. SSH into the affected node. Run df -h to identify "
     "which filesystem is filling up. Use du -sh to find large directories. "
     "Common culprits: container logs in /var/log/pods, unused container images, "
     "and coredump files. Prune unused images with crictl rmi --prune. "
     "Set up log rotation if /var/log is the issue. "
     "If /var/lib/kubelet is filling, check for orphaned volumes or large emptyDir mounts."),
    # 20
    ("NodeMemoryHighUtilization",
     "Alert: NodeMemoryHighUtilization. Node memory utilization is above 90%. "
     "Identify top memory-consuming pods with kubectl top pods --sort-by=memory. "
     "Check if any pods are near their memory limits and at risk of OOMKill. "
     "Consider evicting or rescheduling memory-heavy pods to less loaded nodes. "
     "Check for memory leaks in long-running pods by comparing current vs historical "
     "memory usage. If the node itself is undersized for the workload, "
     "consider upgrading the instance type or adding nodes to the cluster."),
    # 21
    ("NodeCPUSaturation",
     "Alert: NodeCPUSaturation. CPU run queue saturation is high on a node, meaning "
     "more processes are waiting for CPU than can be served. Identify high-CPU pods "
     "with kubectl top pods on the affected node. Check if a batch job or "
     "a recently deployed workload is responsible for the spike. "
     "CPU throttling does not cause saturation directly, saturation means requests "
     "exceed capacity. Consider adding CPU limits to misbehaving pods. "
     "Horizontal scaling of the affected workload may be the fastest mitigation."),
    # 22
    ("NodeNetworkReceiveErrs",
     "Alert: NodeNetworkReceiveErrs. The node is experiencing network receive errors. "
     "Check dmesg for NIC errors or driver messages. Inspect interface statistics "
     "with ip -s link show. Errors can indicate hardware issues, misconfigured MTU, "
     "or network congestion causing packet drops. Check if the issue is isolated "
     "to one interface or affects all. For cloud instances, check the provider's "
     "network health dashboard. MTU mismatches between overlay network and underlying "
     "physical network are a common cause in Kubernetes clusters."),
    # 23
    ("EtcdHighCommitDurations",
     "Alert: EtcdHighCommitDurations. etcd commit durations are elevated, "
     "which will slow down the Kubernetes API server. Check disk I/O on etcd nodes, "
     "etcd is extremely sensitive to disk latency. Run iostat -x 1 to monitor disk. "
     "etcd should be on SSD storage. Check for noisy neighbor processes consuming "
     "I/O on the same host. Check etcd metrics for db_size, large etcd databases "
     "have higher commit latency. Compact and defragment etcd if the database is large: "
     "etcdctl compact and etcdctl defrag. Ensure etcd members are healthy with "
     "etcdctl endpoint health."),
    # 24
    ("EtcdMemberCommunicationSlow",
     "Alert: EtcdMemberCommunicationSlow. etcd member-to-member communication is slow. "
     "Check network latency between etcd members. etcd requires low-latency, "
     "high-bandwidth networking between members, ideally under 10ms RTT. "
     "Check for packet loss between etcd hosts. Verify firewall rules allow "
     "etcd peer traffic on port 2380. If members are in different availability zones, "
     "high cross-AZ latency may be causing this, etcd should generally be co-located "
     "within a region with consistent low-latency networking."),
    # 25
    ("KubeVersionMismatch",
     "Alert: KubeVersionMismatch. Different semantic versions of Kubernetes components "
     "are running in the cluster. This may indicate a failed or partial upgrade. "
     "Check component versions with kubectl version and kubectl get nodes. "
     "Kubernetes supports a skew of one minor version between components. "
     "If the skew exceeds this, upgrade or downgrade components to align versions. "
     "Mixed versions during a rolling upgrade are expected and transient, "
     "this alert should resolve once the upgrade completes. If it persists, "
     "identify which node or component is stuck on the old version."),
    # 26
    ("KubeQuotaAlmostFull",
     "Alert: KubeQuotaAlmostFull. A namespace is approaching its resource quota limit. "
     "Identify which resource is near the limit with kubectl describe quota -n <ns>. "
     "Common resources: CPU requests, memory requests, pod count, PVC count. "
     "If legitimate workloads are growing, request a quota increase from the platform "
     "team. If usage is from stale or orphaned resources, clean them up. "
     "Check for runaway controllers creating excessive pods or configmaps. "
     "Set up alerts earlier in the quota consumption curve to give more reaction time."),
    # 27
    ("KubeHpaMaxedOut",
     "Alert: KubeHpaMaxedOut. A HorizontalPodAutoscaler has been running at its max "
     "replica count for more than 15 minutes. This means the HPA cannot scale further "
     "but demand continues to exceed capacity. Increase the HPA maxReplicas if the "
     "workload legitimately needs more replicas. Check if the underlying metric driving "
     "scaling is still elevated. "
     "Investigate whether a traffic spike is expected or anomalous. "
     "Check cluster capacity, there may be no free nodes to schedule additional pods."),
    # 28
    ("KubeDaemonSetRolloutStuck",
     "Alert: KubeDaemonSetRolloutStuck. A DaemonSet rollout is not making progress. "
     "Check kubectl rollout status daemonset/<name> for details. "
     "Inspect pods that are not yet updated with kubectl get pods -l <selector>. "
     "A common cause is a node that is cordoned or has taints that prevent scheduling. "
     "Check node conditions with kubectl describe node. "
     "If a pod on a specific node is stuck in Pending, check node taints and tolerations. "
     "If pods are crashing after update, roll back with kubectl rollout undo daemonset."),
    # 29
    ("KubeConfigMapSpam",
     "Alert: KubeConfigMapSpam. An unusually high number of ConfigMaps are being "
     "created in a namespace. This is often caused by a controller with a bug that "
     "creates a new ConfigMap on every reconciliation loop instead of updating an "
     "existing one. Identify the creating controller by checking ConfigMap ownerReferences. "
     "Patch or restart the misbehaving controller. Clean up orphaned ConfigMaps. "
     "Large numbers of ConfigMaps slow down API server list operations and etcd."),
]

DOCS_1 = [content for _, content in CORPUS_1]

QUERIES_1 = [
    {"q": "pod crash looping exit code 137",                    "rel": {0}},
    {"q": "pod not ready readiness probe failing",              "rel": {1}},
    {"q": "node NotReady kubelet down",                         "rel": {2}},
    {"q": "deployment replica mismatch rollback",               "rel": {4}},
    {"q": "statefulset PVC pending storage class",              "rel": {5}},
    {"q": "cron job failed backoff limit exceeded",             "rel": {6}},
    {"q": "CPU overcommit node failure tolerance",              "rel": {7}},
    {"q": "memory overcommit OOMKill risk",                     "rel": {8}},
    {"q": "ImagePullBackOff container waiting",                 "rel": {9}},
    {"q": "persistent volume filling up expand PVC",            "rel": {10}},
    {"q": "PV failed detached CSI driver",                      "rel": {11}},
    {"q": "API server 5xx error rate etcd",                     "rel": {12}},
    {"q": "API server latency high etcd slow",                  "rel": {13}},
    {"q": "prometheus not connected alertmanager",              "rel": {15}},
    {"q": "prometheus rule evaluation failures PromQL",         "rel": {16}},
    {"q": "alertmanager config reload failed YAML",             "rel": {17}},
    {"q": "scrape target down ServiceMonitor",                  "rel": {18}},
    {"q": "node filesystem almost full container logs",         "rel": {19}},
    {"q": "node memory utilization 90 percent OOMKill",         "rel": {20}},
    {"q": "etcd commit duration high disk latency",             "rel": {23}},
    {"q": "etcd member communication slow network latency",     "rel": {24}},
    {"q": "kubernetes version mismatch upgrade",                "rel": {25}},
    {"q": "namespace quota almost full pod count",              "rel": {26}},
    {"q": "HPA maxed out max replicas",                         "rel": {27}},
    {"q": "daemonset rollout stuck cordoned node",              "rel": {28}},
    # Paraphrased
    {"q": "container keeps restarting and dying",               "rel": {0}},
    {"q": "worker node went offline",                           "rel": {2, 3}},
    {"q": "not enough pods running for my service",             "rel": {4}},
    {"q": "disk space running out on server",                   "rel": {19}},
    {"q": "cluster database is too slow",                       "rel": {23, 24}},
    {"q": "too many apps scheduled more than cluster can handle","rel": {7, 8}},
    {"q": "autoscaler hit ceiling but load is still high",      "rel": {27}},
    {"q": "alerts not firing when they should",                 "rel": {15}},
    {"q": "storage volume not working after cloud change",      "rel": {11}},
    {"q": "controller creating too many objects in a loop",     "rel": {29}},
]


# ─────────────────────────────────────────────────────────────────────────────
# CORPUS 2 — Kubernetes official troubleshooting documentation
# Source: https://kubernetes.io/docs/tasks/debug/ (Apache 2.0)
# ─────────────────────────────────────────────────────────────────────────────
CORPUS_2 = [
    # 0
    ("DebugPodCrashLooping",
     "Debugging a pod that is crash looping. Use kubectl describe pod to check the Events section "
     "and the Last State field showing the previous container exit code. Use kubectl logs --previous "
     "to retrieve logs from the crashed container. Exit code 137 means OOMKilled. Exit code 1 usually "
     "means an application error. Check resource limits if OOMKilled. Check startup configuration "
     "if the exit code comes from the application itself."),
    # 1
    ("DebugPodPending",
     "Debugging a pod stuck in Pending state. Run kubectl describe pod to check the Events section. "
     "Common causes: insufficient CPU or memory on available nodes, node selector or affinity rules "
     "that cannot be satisfied, PersistentVolumeClaim not bound, resource quota exhausted in the "
     "namespace. Use kubectl get nodes to check node availability and kubectl describe node to check "
     "allocatable resources."),
    # 2
    ("DebugPodImagePullError",
     "Debugging ImagePullBackOff or ErrImagePull errors. The kubelet cannot pull the container image. "
     "Check the image name and tag for typos. Verify the image exists in the registry. For private "
     "registries, ensure the imagePullSecret is correctly configured and the secret exists in the "
     "same namespace as the pod. Use kubectl describe pod to see the exact error message from the "
     "image pull attempt."),
    # 3
    ("DebugServiceNotReachable",
     "Debugging a service that is not reachable. First verify the service exists with kubectl get svc. "
     "Check that the service selector matches the pod labels using kubectl describe svc. Verify the "
     "pod is running and ready. Use kubectl exec to run a shell in a pod and test connectivity with "
     "curl or wget. Check that the targetPort in the service spec matches the containerPort in the "
     "pod spec. Test DNS resolution from within a pod using nslookup."),
    # 4
    ("DebugDNSResolution",
     "Debugging DNS resolution failures in a cluster. Run a test pod with kubectl run to execute "
     "nslookup or dig commands. Check that the CoreDNS pods are running in kube-system namespace. "
     "Inspect CoreDNS logs with kubectl logs. Verify the cluster DNS service is running on the "
     "expected IP. Check /etc/resolv.conf inside the pod. Common issues: CoreDNS ConfigMap "
     "misconfiguration, network policy blocking DNS traffic on port 53, or ndots configuration."),
    # 5
    ("DebugNodeNotReady",
     "Debugging a node in NotReady state. SSH into the node and check kubelet status with "
     "systemctl status kubelet. View kubelet logs with journalctl -u kubelet -n 100. Check disk "
     "pressure with df -h. Check memory pressure with free -m. Check network connectivity. "
     "Check container runtime status. If the node disk is full, clean up unused container images "
     "with crictl rmi --prune. Restart kubelet with systemctl restart kubelet after resolving."),
    # 6
    ("DebugOOMKilled",
     "Debugging OOMKilled containers. The container exceeded its memory limit and was killed by "
     "the kernel. Check the memory limit set in the container spec. Use kubectl top pod to see "
     "current memory consumption. Review application memory usage patterns. Increase the memory "
     "limit if the application legitimately needs more memory. Check for memory leaks if usage "
     "grows unboundedly over time. Consider setting requests equal to limits for memory to avoid "
     "overcommitment."),
    # 7
    ("DebugPodEvicted",
     "Debugging evicted pods. Pods are evicted when a node is under resource pressure. Check "
     "eviction reason with kubectl describe pod. Common eviction conditions: disk pressure, "
     "memory pressure, or PID pressure. Check node conditions with kubectl describe node. "
     "Evicted pods leave behind their definition but do not restart automatically. To prevent "
     "eviction, set appropriate resource requests so the scheduler places pods correctly, "
     "and configure PodDisruptionBudgets."),
    # 8
    ("DebugNetworkPolicy",
     "Debugging network policy issues. If pods cannot communicate, a NetworkPolicy may be "
     "blocking traffic. Use kubectl get networkpolicy to list policies in the namespace. "
     "Check policy selectors against pod labels. Test connectivity with kubectl exec and curl. "
     "A common mistake is creating an overly restrictive ingress policy without a matching "
     "egress policy. Check that the CNI plugin installed supports NetworkPolicy enforcement. "
     "Calico, Cilium, and Weave Net enforce NetworkPolicy; Flannel does not by default."),
    # 9
    ("DebugConfigMapMount",
     "Debugging ConfigMap or Secret mount issues. If a pod fails to start with a "
     "CreateContainerConfigError, the referenced ConfigMap or Secret does not exist. "
     "Verify with kubectl get configmap or kubectl get secret in the correct namespace. "
     "Check that the key names referenced in the pod spec match the keys in the ConfigMap. "
     "For volume mounts, check that the mountPath does not conflict with existing directories "
     "in the container image. After fixing the ConfigMap, delete and recreate the pod."),
    # 10
    ("DebugPersistentVolume",
     "Debugging PersistentVolume binding failures. If a PVC is stuck in Pending, check available "
     "PersistentVolumes with kubectl get pv. Verify the PVC access mode and storage class match "
     "an available PV. Check that the requested storage size does not exceed available PV capacity. "
     "For dynamic provisioning, verify the StorageClass exists and the provisioner pod is running. "
     "Check provisioner logs for errors. If the PVC was previously bound, check if the PV was "
     "deleted or its reclaim policy caused it to be released."),
    # 11
    ("DebugHorizontalPodAutoscaler",
     "Debugging HorizontalPodAutoscaler not scaling. Check HPA status with kubectl describe hpa. "
     "Verify the metrics server is installed and running. Use kubectl top pods to confirm metrics "
     "are available. Check if the current replica count is already at maxReplicas. Verify the "
     "target metric name matches what the application exposes. Check for ScalingLimited conditions "
     "in the HPA status. Custom metrics require the custom metrics API server to be deployed."),
    # 12
    ("DebugInitContainer",
     "Debugging init container failures. Init containers run before the main container and must "
     "complete successfully. Use kubectl describe pod to see init container status. Use "
     "kubectl logs <pod> -c <init-container-name> to view init container logs. Common causes: "
     "database not yet ready, dependency service unreachable, or misconfigured environment "
     "variables. Init containers run in order and each must exit with code 0 before the next "
     "starts. If an init container keeps restarting, the pod will show Init:CrashLoopBackOff."),
    # 13
    ("DebugRBACPermissions",
     "Debugging RBAC permission errors. If a pod gets a Forbidden error calling the Kubernetes "
     "API, the service account lacks the required permissions. Check the service account with "
     "kubectl describe pod. Use kubectl auth can-i to test permissions. Check ClusterRoleBindings "
     "and RoleBindings for the service account. Create or update a Role or ClusterRole and bind "
     "it to the service account. Be specific with permissions: avoid cluster-admin unless necessary."),
    # 14
    ("DebugCertificateExpiry",
     "Debugging TLS certificate expiry in a cluster. Expired certificates cause API server "
     "authentication failures. Check certificate expiry with kubeadm certs check-expiration. "
     "Renew certificates with kubeadm certs renew all. After renewing, restart the API server, "
     "controller manager, and scheduler. For certificates managed outside kubeadm, use openssl "
     "x509 -in cert.pem -noout -dates to check expiry. Set up monitoring alerts for certificates "
     "expiring within 30 days."),
    # 15
    ("DebugEtcdHealth",
     "Debugging etcd health issues. A degraded etcd cluster causes API server slowness and "
     "failures. Check etcd member health with etcdctl endpoint health. Check etcd metrics for "
     "high fsync latency. etcd is sensitive to disk latency and should run on SSD storage. "
     "Check disk I/O with iostat. Defragment etcd if database size is large with etcdctl defrag. "
     "Check the number of objects in etcd: a very large etcd database degrades performance. "
     "Compact old revisions with etcdctl compact."),
    # 16
    ("DebugAPIServerSlowness",
     "Debugging slow Kubernetes API server responses. Check API server metrics for request "
     "latency. High latency often correlates with etcd slowness. Check etcd disk I/O. Look for "
     "expensive list operations in audit logs. Enable API Priority and Fairness to rate-limit "
     "expensive requests. Check for runaway controllers making excessive API calls. Monitor "
     "apiserver_request_duration_seconds metrics. Increase API server resources if it is CPU "
     "or memory constrained."),
    # 17
    ("DebugJobNotCompleting",
     "Debugging a Kubernetes Job that is not completing. Check job status with kubectl describe job. "
     "View pod logs from job pods. Check if backoffLimit has been reached. If the job creates "
     "pods that keep failing, the job will eventually stop retrying. Check for resource constraints "
     "preventing pods from being scheduled. For CronJobs, check the last schedule time and whether "
     "the previous job is still running and concurrencyPolicy prevents new runs."),
    # 18
    ("DebugIngressNotWorking",
     "Debugging Ingress not routing traffic. Verify the Ingress controller pod is running in the "
     "cluster. Check Ingress resource with kubectl describe ingress. Verify the backend service "
     "and port match a running service. Check Ingress controller logs for routing errors. "
     "Verify TLS secret exists if TLS is configured. Test the backend service directly with "
     "kubectl port-forward to isolate whether the issue is in the Ingress or the service. "
     "Check annotations required by the specific Ingress controller being used."),
    # 19
    ("DebugContainerRuntimeIssue",
     "Debugging container runtime issues. If pods fail to start with a ContainerCannotRun error, "
     "the container runtime may have an issue. Check containerd or CRI-O status on the node with "
     "systemctl status containerd. Check runtime logs with journalctl -u containerd. Verify the "
     "socket path is correctly configured in kubelet. Check disk space on the node as a full disk "
     "can prevent containers from starting. Restart the container runtime if necessary."),
    # 20
    ("DebugStatefulSetStorage",
     "Debugging StatefulSet storage issues. StatefulSets create PVCs per replica. If a PVC is "
     "lost, the StatefulSet pod will not start. Check PVC status with kubectl get pvc. If the "
     "storage backend was deleted, you may need to manually create a PV and PVC to restore the "
     "original claim. StatefulSets do not automatically recreate PVCs. "
     "Check that the StorageClass allows volume expansion if you need to resize. "
     "PVC names follow the pattern <volumeClaimTemplate-name>-<pod-name>."),
    # 21
    ("DebugResourceQuotaExceeded",
     "Debugging resource quota exceeded errors. If a pod fails to schedule with a quota error, "
     "check namespace quotas with kubectl describe quota. Identify which resource is exhausted: "
     "CPU, memory, pod count, or service count. Delete unused resources to free quota. Request "
     "a quota increase from the cluster administrator. Check LimitRange objects that may be "
     "injecting default resource requests higher than expected."),
    # 22
    ("DebugDaemonSetNotScheduling",
     "Debugging a DaemonSet pod not scheduling on a node. DaemonSets schedule one pod per node "
     "unless a node is tainted or unschedulable. Check node taints with kubectl describe node. "
     "Add a toleration to the DaemonSet spec if the node has a taint the DaemonSet should tolerate. "
     "Check if the node is cordoned with kubectl get nodes. Check nodeSelector in the DaemonSet "
     "spec. If the DaemonSet pod is in Pending state, check resource availability on that node."),
    # 23
    ("DebugMultiContainerPod",
     "Debugging issues in a multi-container pod. Each container in a pod shares the network "
     "namespace but has its own filesystem. Use kubectl logs <pod> -c <container> to view logs "
     "from a specific container. Use kubectl exec <pod> -c <container> to get a shell in a "
     "specific container. If containers communicate over localhost, check that the correct port "
     "is used. Check that sidecar containers are not consuming excessive resources and starving "
     "the main container."),
    # 24
    ("DebugProbeFailures",
     "Debugging liveness and readiness probe failures. Probe failures cause pods to be restarted "
     "liveness or removed from service endpoints readiness. Check probe configuration in "
     "kubectl describe pod. Common issues: incorrect path or port, insufficient initialDelaySeconds "
     "for slow-starting applications, probe timeout too short, or the health endpoint returning "
     "non-200 for legitimate reasons. Temporarily increase failureThreshold to prevent restarts "
     "while debugging. Check application logs around the time probes fail."),
    # 25
    ("DebugSchedulerNotPlacing",
     "Debugging pods not being placed by the scheduler. If a pod is stuck in Pending with no "
     "Events from the scheduler, check if the scheduler pod is running in kube-system. Check "
     "scheduler logs for errors. Verify node affinity and anti-affinity rules are not too "
     "restrictive. Check for taints and tolerations mismatches. Use kubectl describe pod to see "
     "scheduler events. Run kubectl get events --field-selector reason=FailedScheduling "
     "to list recent scheduling failures."),
    # 26
    ("DebugCoreDNSCrashing",
     "Debugging CoreDNS pod crashes. CoreDNS failures cause cluster-wide DNS resolution failures. "
     "Check CoreDNS pod status in kube-system namespace. View CoreDNS logs. Check the CoreDNS "
     "ConfigMap for syntax errors. Common issues: upstream DNS server unreachable, loop detection "
     "triggering on clusters where the node DNS points to CoreDNS itself, or insufficient "
     "resources. Increase CoreDNS memory limits if it is OOMKilled. Check for DNS amplification "
     "by high-volume pods querying external domains."),
    # 27
    ("DebugMutatingWebhook",
     "Debugging mutating and validating admission webhooks blocking pod creation. If pods fail "
     "with an error from a webhook, identify the webhook with kubectl get mutatingwebhookconfiguration. "
     "Check if the webhook service is running and reachable. Webhooks with failurePolicy: Fail "
     "will block all pod creation if the webhook service is down. Temporarily set failurePolicy "
     "to Ignore to restore cluster operation while fixing the webhook. Check webhook TLS "
     "certificate validity. Review webhook logs for the rejection reason."),
    # 28
    ("DebugNodeDiskPressure",
     "Debugging node disk pressure evictions. When a node runs low on disk, the kubelet evicts "
     "pods to free space. Check disk usage on the node with df -h. Large consumers are usually "
     "container logs, unused images, and temporary files. Prune unused images with "
     "crictl rmi --prune. Configure log rotation to prevent log files from growing unboundedly. "
     "Set eviction thresholds in kubelet configuration. Consider adding more disk capacity if "
     "the workload legitimately requires it."),
    # 29
    ("DebugPortForwardFailing",
     "Debugging kubectl port-forward failures. Port-forward fails if the pod is not running, "
     "the port is not open in the container, or the connection times out. Check pod status first. "
     "Verify the containerPort in the pod spec matches the port being forwarded. If port-forward "
     "connects but immediately closes, the application may not be listening on that port. "
     "Use kubectl exec to check what ports are open inside the container with ss -tlnp. "
     "Port-forward goes through the API server so API server health affects it."),
]

DOCS_2 = [content for _, content in CORPUS_2]

QUERIES_2 = [
    {"q": "pod crash looping OOMKilled exit code 137",           "rel": {0, 6}},
    {"q": "pod stuck in Pending node affinity",                  "rel": {1}},
    {"q": "ImagePullBackOff private registry imagePullSecret",   "rel": {2}},
    {"q": "service not reachable selector labels targetPort",    "rel": {3}},
    {"q": "DNS resolution failure CoreDNS nslookup",             "rel": {4}},
    {"q": "node NotReady kubelet disk pressure",                 "rel": {5}},
    {"q": "container OOMKilled memory limit exceeded",           "rel": {6}},
    {"q": "pod evicted node memory pressure",                    "rel": {7}},
    {"q": "NetworkPolicy blocking traffic CNI Calico Cilium",    "rel": {8}},
    {"q": "ConfigMap Secret CreateContainerConfigError",         "rel": {9}},
    {"q": "PVC pending StorageClass dynamic provisioning",       "rel": {10}},
    {"q": "HPA not scaling metrics server custom metrics",       "rel": {11}},
    {"q": "init container CrashLoopBackOff dependency",          "rel": {12}},
    {"q": "RBAC Forbidden service account permissions",          "rel": {13}},
    {"q": "TLS certificate expiry kubeadm renew",                "rel": {14}},
    {"q": "etcd health fsync latency defrag compact",            "rel": {15}},
    {"q": "API server slow latency audit logs",                  "rel": {16}},
    {"q": "Job not completing backoffLimit CronJob",             "rel": {17}},
    {"q": "Ingress not routing TLS secret controller logs",      "rel": {18}},
    {"q": "containerd CRI socket kubelet runtime",               "rel": {19}},
    {"q": "StatefulSet PVC lost storage deleted",                "rel": {20}},
    {"q": "resource quota exceeded namespace pod count",         "rel": {21}},
    {"q": "DaemonSet not scheduling node taint toleration",      "rel": {22}},
    {"q": "liveness readiness probe failure initialDelaySeconds","rel": {24}},
    {"q": "scheduler Pending FailedScheduling affinity taint",   "rel": {25}},
    {"q": "CoreDNS crashing loop detection OOMKilled",           "rel": {26}},
    {"q": "webhook failurePolicy blocking pod creation",         "rel": {27}},
    {"q": "node disk pressure log rotation crictl prune",        "rel": {28}},
    # Paraphrased
    {"q": "my container keeps dying and restarting",             "rel": {0}},
    {"q": "pods not starting on a specific node",                "rel": {1, 5, 22}},
    {"q": "cannot pull docker image from private repo",          "rel": {2}},
    {"q": "app inside cluster cannot talk to another service",   "rel": {3, 8}},
    {"q": "cluster DNS is broken nothing resolves",              "rel": {4, 26}},
    {"q": "worker node went offline",                            "rel": {5}},
    {"q": "app running out of memory",                           "rel": {6}},
    {"q": "certificates expired cluster not working",            "rel": {14}},
]


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def precision_at_k(retrieved, relevant, k):
    return sum(1 for r in retrieved[:k] if r in relevant) / k

def reciprocal_rank(retrieved, relevant):
    for rank, idx in enumerate(retrieved, 1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0

def evaluate(name, retrieve_fn, queries, k_values=(1, 3, 5)):
    pk = {k: [] for k in k_values}
    mrr, lats = [], []
    for item in queries:
        t0 = time.perf_counter()
        ranked = retrieve_fn(item["q"])
        lats.append((time.perf_counter() - t0) * 1000)
        mrr.append(reciprocal_rank(ranked, item["rel"]))
        for k in k_values:
            pk[k].append(precision_at_k(ranked, item["rel"], k))
    return {
        "name": name,
        "num_queries": len(queries),
        "MRR":            round(float(np.mean(mrr)), 4),
        "P@1":            round(float(np.mean(pk[1])), 4),
        "P@3":            round(float(np.mean(pk[3])), 4),
        "P@5":            round(float(np.mean(pk[5])), 4),
        "latency_p50_ms": round(float(np.percentile(lats, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(lats, 95)), 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BM25
# ─────────────────────────────────────────────────────────────────────────────
def build_bm25(docs):
    return BM25Okapi([d.lower().split() for d in docs])

def bm25_retrieve(bm25, query, top_k=10):
    scores = bm25.get_scores(query.lower().split())
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# VECTOR
# ─────────────────────────────────────────────────────────────────────────────
def build_vector_index(docs):
    model = SentenceTransformer(VECTOR_MODEL)
    t0 = time.perf_counter()
    embeddings = model.encode(docs, show_progress_bar=False).astype(np.float64)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return model, embeddings, elapsed_ms

def vector_retrieve(query, model, embeddings, top_k=10):
    q_emb = model.encode([query]).astype(np.float64)
    sims = cosine_similarity(q_emb, embeddings)[0]
    return sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# RUN ONE CORPUS
# ─────────────────────────────────────────────────────────────────────────────
def run_corpus(label, docs, queries, model, shared_model=None):
    print(f"\n{'='*62}")
    print(f"{label}")
    print(f"Docs: {len(docs)}  Queries: {len(queries)}")
    print(f"{'='*62}")

    # BM25
    print("\n[BM25]")
    t0 = time.perf_counter()
    bm25 = build_bm25(docs)
    idx_ms = (time.perf_counter() - t0) * 1000
    print(f"  Index built in {idx_ms:.2f}ms")
    bm25_res = evaluate("BM25", lambda q: bm25_retrieve(bm25, q), queries)
    bm25_res["index_build_ms"] = round(idx_ms, 3)

    # Vector — reuse loaded model across corpora
    print(f"\n[Vector — {VECTOR_MODEL}]")
    if shared_model is None:
        m, emb, vec_ms = build_vector_index(docs)
    else:
        m = shared_model
        t0 = time.perf_counter()
        emb = m.encode(docs, show_progress_bar=False).astype(np.float64)
        vec_ms = (time.perf_counter() - t0) * 1000
    print(f"  Index built in {vec_ms:.2f}ms")
    vec_res = evaluate(f"Vector ({VECTOR_MODEL})",
                       lambda q: vector_retrieve(q, m, emb), queries)
    vec_res["index_build_ms"] = round(vec_ms, 3)

    # Print
    print(f"\n{'Metric':<26} {'BM25':>12} {'Vector':>12}")
    print("-" * 52)
    for metric in ["P@1", "P@3", "P@5", "MRR",
                   "latency_p50_ms", "latency_p95_ms", "index_build_ms"]:
        print(f"{metric:<26} {str(bm25_res[metric]):>12} {str(vec_res[metric]):>12}")

    return bm25_res, vec_res, m


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Retrieval Evaluation: BM25 vs Vector Search")
    print("Two corpora, one script.")

    bm25_c1, vec_c1, model = run_corpus(
        "Corpus 1: Prometheus Operator Runbooks (Apache 2.0)\n"
        "Source: https://github.com/prometheus-operator/runbooks",
        DOCS_1, QUERIES_1, VECTOR_MODEL
    )

    bm25_c2, vec_c2, _ = run_corpus(
        "Corpus 2: Kubernetes Troubleshooting Docs (Apache 2.0)\n"
        "Source: https://kubernetes.io/docs/tasks/debug/",
        DOCS_2, QUERIES_2, VECTOR_MODEL, shared_model=model
    )

    # Save JSON
    output = {
        "vector_model": VECTOR_MODEL,
        "corpus_1": {
            "name": "Prometheus Operator runbooks",
            "source": "https://github.com/prometheus-operator/runbooks",
            "license": "Apache 2.0",
            "num_docs": len(DOCS_1),
            "num_queries": len(QUERIES_1),
            "bm25": bm25_c1,
            "vector": vec_c1,
        },
        "corpus_2": {
            "name": "Kubernetes troubleshooting documentation",
            "source": "https://kubernetes.io/docs/tasks/debug/",
            "license": "Apache 2.0",
            "num_docs": len(DOCS_2),
            "num_queries": len(QUERIES_2),
            "bm25": bm25_c2,
            "vector": vec_c2,
        },
    }
    with open("eval_unified_results.json", "w") as f:
        json.dump(output, f, indent=2)

    # Save Markdown
    def tbl(c):
        b, v = c["bm25"], c["vector"]
        return f"""| Metric | BM25 | Vector ({VECTOR_MODEL}) |
|--------|------|--------------------------|
| P@1 | {b['P@1']} | {v['P@1']} |
| P@3 | {b['P@3']} | {v['P@3']} |
| P@5 | {b['P@5']} | {v['P@5']} |
| MRR | {b['MRR']} | {v['MRR']} |
| Query latency p50 (ms) | {b['latency_p50_ms']} | {v['latency_p50_ms']} |
| Query latency p95 (ms) | {b['latency_p95_ms']} | {v['latency_p95_ms']} |
| Index build (ms) | {b['index_build_ms']} | {v['index_build_ms']} |"""

    md = f"""# Retrieval Evaluation Results

## Corpus 1: Prometheus Operator Runbooks
**Source:** https://github.com/prometheus-operator/runbooks (Apache 2.0)
**Docs:** {len(DOCS_1)}  **Queries:** {len(QUERIES_1)}

{tbl(output['corpus_1'])}

## Corpus 2: Kubernetes Troubleshooting Documentation
**Source:** https://kubernetes.io/docs/tasks/debug/ (Apache 2.0)
**Docs:** {len(DOCS_2)}  **Queries:** {len(QUERIES_2)}

{tbl(output['corpus_2'])}
"""
    with open("eval_unified_results.md", "w") as f:
        f.write(md)

    print("\nSaved: eval_unified_results.json")
    print("Saved: eval_unified_results.md")
    print("\nDone.")
