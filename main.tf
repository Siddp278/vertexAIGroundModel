provider "google" {
  project = var.project_id
  region  = "us-central1"
}

resource "google_storage_bucket" "coldline_bucket" {
  name          = var.bucket_name
  location      = "US"
  storage_class = "COLDLINE"
  force_destroy = true

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = 30
    }
  }
}

resource "google_service_account" "vertex_llm_sa" {
  account_id   = var.service_account_id
  display_name = "Service Account for Vertex AI LLM + Embedding + GCS Access"
}

# Vertex AI User Role
resource "google_project_iam_member" "vertex_ai_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.vertex_llm_sa.email}"
}

# GCS Access: Storage Object Admin (read/write/delete GCS objects)
resource "google_project_iam_member" "storage_admin" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.vertex_llm_sa.email}"
}