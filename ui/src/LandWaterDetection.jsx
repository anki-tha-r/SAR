// ...existing code...
import React, { useState, useRef} from "react";
import './UploadFolder.css';
import { useLocation } from 'react-router-dom';

const LandWaterDetection = () => {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [outputs, setOutputs] = useState([]); // <- new
  const inputRef = useRef(null);
  const location = useLocation();
  const singleInputRef = useRef(null);

const onSelectImage = (e) => {
   const f = e?.target?.files && e.target.files[0];
    if (f && f.type && f.type.startsWith("image/")) {
      setFiles([f]);
      setStatus(null);
      setProgress(0);
    } else {
      setFiles([]);
    }
  };

    const fetchOutputs = async () => {
      const pagePath = encodeURIComponent(location.pathname || '');
      window.location.href = `http://localhost:8001/outputs_landwater`;
  };

  const handleFiles = (fileList) => {
    const arr = Array.from(fileList).filter(f => f.type.startsWith("image/"));
    setFiles(arr);
    setStatus(null);
    setProgress(0);
  };

  const onSelectFolder = (e) => {
    handleFiles(e.target.files);
  };

  const onDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer?.files) handleFiles(e.dataTransfer.files);
  };

  const onDragOver = (e) => {
    e.preventDefault();
  };

  const uploadFiles = () => {
    if (!files.length) return;
    setUploading(true);
    setStatus(null);
    setProgress(0);

    const form = new FormData();
    files.forEach((file) => {
      // preserve folder structure when available (webkitRelativePath)
      const key = file.webkitRelativePath || file.name;
      form.append("images", file, key);
    });

    // Use XMLHttpRequest to track upload progress
    const xhr = new XMLHttpRequest();
    const pagePath = encodeURIComponent(location.pathname || '');
    xhr.open("POST", `http://localhost:8001/upload`, true);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 100);
        setProgress(pct);
      }
    };

    xhr.onload = () => {
      setUploading(false);
      if (xhr.status >= 200 && xhr.status < 300) {
        setStatus({ ok: true, message: "Upload complete. Processing images..." });
        setFiles([]);
        if (inputRef.current) inputRef.current.value = null;
      } else {
        setStatus({ ok: false, message: `Upload failed (${xhr.status})` });
      }
    };

    xhr.onerror = () => {
      setUploading(false);
      setStatus({ ok: false, message: "Upload error" });
    };

    xhr.send(form);
    // refresh outputs list after successful processing
    try {
      // call processing for the first uploaded file (or loop files if you want each processed)
      if (files.length > 0) postSingleFileToLand(files[0]);
      else postSingleFileToLand(); // fallback (server may handle latest upload)
    } catch (e) { /* ignore fetchOutputs errors */ }
  };

        const process_image = async () => {
            setProcessing(true);
            setStatus(null);

            try {
                const resp = await fetch("http://localhost:8001/process_landwater", { method: "POST" });
                if (!resp.ok) throw new Error(`Server returned ${resp.status}`);
                const payload = await resp.json();

                setStatus({ ok: true, message: "Processing complete — results ready to view." });
            } catch (err) {
                setStatus({ ok: false, message: `Processing failed: ${err.message}` });
            } finally {
                setProcessing(false);
            }
        };

    // ...inside your component...
const postSingleFileToLand = async (file) => {
  const fd = new FormData();
  fd.append('file', file, file.name);
  const resp = await fetch('http://127.0.0.1:8001/process_landwater', {
    method: 'POST',
    body: fd, // DO NOT set Content-Type header explicitly
  });
  const data = await resp.json();
  console.log('process_landwater response', resp.status, data);
};

  return (
    <div style={{ fontFamily: "Inter, system-ui, sans-serif", maxWidth: 900, margin: "32px auto", padding: 20 }}>
      <header style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 18 }}>
        <div aria-hidden="true" style={{ width: 44, height: 44, display: "flex", alignItems: "center", justifyContent: "center", background: "#eef2ff", borderRadius: 8 }}>
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="9" stroke="#2dd4bf" strokeWidth="1.2" fill="none" />
            <path d="M5 15c2-3 6-5 8-5s6 2 8 5" stroke="#6366f1" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" fill="none" />
          </svg>
        </div>
        <div>
          <div style={{ fontWeight: 700 }}>Spectra Space</div>
          <div style={{ fontSize: 12, color: "#475569" }}>Land & Water Detection — Colorized SAR</div>
        </div>
      </header>

      <h2 style={{ marginTop: 0 }}>Upload folder of images</h2>

      <div
        onDrop={onDrop}
        onDragOver={onDragOver}
        style={{
          border: "2px dashed #c7d2fe",
          borderRadius: 10,
          padding: 18,
          textAlign: "center",
          background: "#fbfbff"
        }}
      >
        <p style={{ margin: "6px 0 12px" }}>Drag & drop a folder here or choose a folder using the button.</p>

        <input
          ref={inputRef}
          type="file"
          webkitdirectory="true"
          directory=""
          multiple
          accept="image/*"
          onChange={onSelectFolder}
          style={{ display: "none" }}
        />

        <div style={{ display: "flex", gap: 8, justifyContent: "center" }}>
          <button
            type="button"
            onClick={() => inputRef.current && inputRef.current.click()}
            style={{ padding: "8px 14px", borderRadius: 8, border: "1px solid #c7d2fe", background: "white", cursor: "pointer" }}
          >
            Select Folder
          </button>

          <button
            type="button"
            onClick={uploadFiles}
            disabled={uploading || files.length === 0}
            style={{
              padding: "8px 14px",
              borderRadius: 8,
              border: "none",
              background: files.length ? "#7c3aed" : "#ddd",
              color: "white",
              cursor: files.length ? "pointer" : "not-allowed"
            }}
          >
            {uploading ? `Uploading... ${progress}%` : "Upload"}
          </button>
        </div>
      </div>

       <h2 style={{ marginTop: 0 }}>Upload a single image</h2>

      <div
        onDrop={onDrop}
        onDragOver={onDragOver}
        style={{
          border: "2px dashed #c7d2fe",
          borderRadius: 10,
          padding: 18,
          textAlign: "center",
          background: "#fbfbff"
        }}
      >
        <p style={{ margin: "6px 0 12px" }}>Drag & drop an image here or choose an image using the button.</p>
        
        <input
          ref={singleInputRef}
          type="file"
          accept="image/*"
          onChange={onSelectImage}
          style={{ display: "none" }}
        />

        <div style={{ display: "flex", gap: 8, justifyContent: "center" }}>
          <button
            type="button"
            onClick={() => singleInputRef.current && singleInputRef.current.click()}
            style={{ padding: "8px 14px", borderRadius: 8, border: "1px solid #c7d2fe", background: "white", cursor: "pointer" }}
          >
            Select Image
          </button>

          <button
            type="button"
            onClick={uploadFiles}
            disabled={uploading || files.length === 0}
            style={{
              padding: "8px 14px",
              borderRadius: 8,
              border: "none",
              background: files.length ? "#7c3aed" : "#ddd",
              color: "white",
              cursor: files.length ? "pointer" : "not-allowed"
            }}
          >
            {uploading ? `Uploading... ${progress}%` : "Upload"}
          </button>
        </div>
      </div>

      {files.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <strong>Selected images ({files.length}):</strong>
          <ul style={{ maxHeight: 220, overflow: "auto", paddingLeft: 16 }}>
            {files.map((f, i) => (
              <li key={i} style={{ margin: "6px 0", display: "flex", gap: 8, alignItems: "center" }}>
                <img
                  src={URL.createObjectURL(f)}
                  alt={f.name}
                  style={{ width: 48, height: 48, objectFit: "cover", borderRadius: 6, border: "1px solid #e6e9f2" }}
                />
                <div style={{ fontSize: 13 }}>
                  <div style={{ fontWeight: 600 }}>{f.webkitRelativePath || f.name}</div>
                  <div style={{ color: "#64748b", fontSize: 12 }}>{Math.round(f.size / 1024)} KB</div>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}

      {uploading && (
        <div style={{ marginTop: 12 }}>
          <div style={{ height: 10, width: "100%", background: "#eee", borderRadius: 6, overflow: "hidden" }}>
            <div style={{ width: `${progress}%`, height: "100%", background: "#7c3aed" }} />
          </div>
        </div>
      )}

      {status && (
        <div style={{ marginTop: 12, color: status.ok ? "#065f46" : "#b91c1c" }}>
          {status.message}
        </div>
         )}

      {(status && !processing)&&(
        <div style={{ marginTop: 12, color: status.ok ? "#065f46" : "#b91c1c" }}>
          <button onClick={fetchOutputs}>View Results</button>
        </div>
      )}
    </div>
  );
};

export default LandWaterDetection;