import { useState } from "react";
import Swal from "sweetalert2";
import "../styles.css";

export function UploadSection() {
  const [file, setFile] = useState(null);
  const [privateKey, setPrivateKey] = useState("");
  const [status, setStatus] = useState("idle"); // idle, uploading, success, error

  const handleFileChange = (e) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file || !privateKey) {
      Swal.fire({
        title: "Campos incompletos",
        text: "Por favor selecciona un archivo y tu llave privada.",
        icon: "warning",
        confirmButtonColor: "#15847d",
      });
      return;
    }

    setStatus("uploading");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/api/cow-data", {
        method: "POST",
        headers: {
          Authorization: privateKey,
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log("Upload success:", data);
      setStatus("success");

      Swal.fire({
        title: "¡Éxito!",
        text: "Archivo subido exitosamente.",
        icon: "success",
        confirmButtonColor: "#15847d",
        timer: 2000,
      });

      setFile(null);
      setPrivateKey("");
    } catch (error) {
      console.error("Upload failed:", error);
      setStatus("error");

      Swal.fire({
        title: "Error",
        text: `Error al subir el archivo: ${error.message}`,
        icon: "error",
        confirmButtonColor: "#d33",
      });
    } finally {
      setStatus("idle");
    }
  };

  return (
    <section className="upload-card">
      <header>
        <div>
          <h2>Subir Registros CSV</h2>
          <p className="upload-subtitle">
            Sube archivos de ordeño diarios con tu llave privada.
          </p>
        </div>
      </header>
      <div className="upload-content">
        <form onSubmit={handleSubmit} className="upload-form">
          <div className="form-group">
            <label htmlFor="csvFile">Archivo CSV</label>
            <div className="file-input-wrapper">
              <input
                type="file"
                id="csvFile"
                accept=".csv"
                onChange={handleFileChange}
                className="file-input"
              />
              <div className="file-input-trigger">
                {file ? file.name : "Seleccionar archivo..."}
              </div>
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="privateKey">Llave Privada</label>
            <input
              type="password"
              id="privateKey"
              value={privateKey}
              onChange={(e) => setPrivateKey(e.target.value)}
              placeholder="Ingresa tu llave de acceso"
              className="text-input"
            />
          </div>

          <button
            type="submit"
            className="primary-btn"
            disabled={status === "uploading"}
          >
            {status === "uploading" ? "Subiendo..." : "Subir Datos"}
          </button>
        </form>
      </div>
    </section>
  );
}
