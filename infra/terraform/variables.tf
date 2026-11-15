variable "tenancy_ocid" {
  type        = string
  description = "OCID of the tenancy (Profile > Tenancy in the OCI console)."
}

variable "user_ocid" {
  type        = string
  description = "OCID of the API user."
}

variable "fingerprint" {
  type        = string
  description = "Fingerprint of the API signing key."
}

variable "private_key_path" {
  type        = string
  description = "Path to the API signing private key (PEM)."
}

variable "region" {
  type        = string
  description = "OCI home region; Always Free A1 capacity exists only in the home region."
  default     = "ap-singapore-1"
}

variable "compartment_ocid" {
  type        = string
  description = "Compartment to create resources in (the tenancy OCID works for a personal account)."
}

variable "ssh_public_key" {
  type        = string
  description = "Public key installed for the ubuntu user."
}

variable "admin_cidr" {
  type        = string
  description = "CIDR allowed to SSH in, e.g. your IP/32. Never 0.0.0.0/0."
  validation {
    condition     = can(cidrnetmask(var.admin_cidr)) && var.admin_cidr != "0.0.0.0/0"
    error_message = "admin_cidr must be a valid CIDR and not 0.0.0.0/0."
  }
}

# Always Free covers 4 OCPU / 24 GB of A1 in total; 2 / 12 leaves headroom for another VM.
variable "ocpus" {
  type    = number
  default = 2
  validation {
    condition     = var.ocpus >= 1 && var.ocpus <= 4
    error_message = "Always Free allows at most 4 A1 OCPUs."
  }
}

variable "memory_gb" {
  type    = number
  default = 12
  validation {
    condition     = var.memory_gb >= 6 && var.memory_gb <= 24
    error_message = "Use 6-24 GB: the API loads torch + two models (~2 GB) next to Postgres."
  }
}

variable "boot_volume_gb" {
  type        = number
  description = "Boot volume size; Always Free includes 200 GB of block storage in total."
  default     = 100
}

variable "repo_url" {
  type        = string
  description = "Git URL of this repository, cloned on first boot."
}

variable "repo_ref" {
  type    = string
  default = "main"
}

variable "postgres_password" {
  type        = string
  sensitive   = true
  description = "Password for the rag Postgres user (container on the VM, never exposed)."
}

variable "anthropic_api_key" {
  type        = string
  sensitive   = true
  default     = ""
  description = "Optional. Empty means the extractive mock provider."
}
