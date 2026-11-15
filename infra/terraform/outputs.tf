output "public_ip" {
  value       = oci_core_instance.app.public_ip
  description = "Web UI at http://<ip>/ once cloud-init has finished (first boot ingests the corpus)."
}

output "ssh" {
  value = "ssh ubuntu@${oci_core_instance.app.public_ip}"
}
