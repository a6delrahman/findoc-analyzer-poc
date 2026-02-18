"""
SharePoint Service - Streamlined for document upload via Microsoft Graph API.
"""
import logging
import requests
from typing import Any, Dict, Optional
from azure.identity import DefaultAzureCredential
from env_config import get_config
import pandas as pd
from io import BytesIO

config = get_config()
logger = logging.getLogger(__name__)

CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
CONTENT_TYPE_PDF = "application/pdf"


class SharePointService:
    """Handles document upload to SharePoint using Microsoft Graph API."""
    
    def __init__(
        self,
        hostname: str = "visionandlab.sharepoint.com",
        site_path: str = "/sites/FinDocAnalyser"
    ):
        self.credential = DefaultAzureCredential()
        self.graph_base_url = "https://graph.microsoft.com/v1.0"
        self.hostname = hostname
        self.site_path = site_path
        self._site_id_cache: Optional[str] = None
    
    def _get_token(self) -> str:
        """Get access token for Microsoft Graph API."""
        token = self.credential.get_token("https://graph.microsoft.com/.default")
        return token.token
    
    def _get_headers(self, content_type: str = CONTENT_TYPE_JSON) -> Dict[str, str]:
        """Get request headers with authorization."""
        return {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": content_type
        }
    
    def get_site_id(self) -> str:
        """Get the SharePoint site ID (cached)."""
        if self._site_id_cache:
            return self._site_id_cache
        
        url = f"{self.graph_base_url}/sites/{self.hostname}:{self.site_path}"
        response = requests.get(url, headers=self._get_headers(), timeout=30)
        response.raise_for_status()
        
        self._site_id_cache = response.json().get("id")
        logger.info(f"Retrieved site ID: {self._site_id_cache}")
        return self._site_id_cache
    
    def ensure_folder_exists(self, folder_path: str) -> bool:
        """
        Ensure the folder path exists, creating folders as needed.
        
        Args:
            folder_path: Path like "UseCase-1" or "UseCase-2"
            
        Returns:
            True if successful
        """
        try:
            site_id = self.get_site_id()
            headers = self._get_headers()
            
            parts = [p for p in folder_path.split('/') if p]
            current_path = ""
            
            for i, part in enumerate(parts):
                current_path = f"{current_path}/{part}" if current_path else part
                
                # Check if folder exists
                check_url = f"{self.graph_base_url}/sites/{site_id}/drive/root:/{current_path}"
                response = requests.get(check_url, headers=headers, timeout=30)
                
                if response.status_code == 404:
                    # Create folder
                    if i == 0:
                        create_url = f"{self.graph_base_url}/sites/{site_id}/drive/root/children"
                    else:
                        parent_path = "/".join(parts[:i])
                        create_url = f"{self.graph_base_url}/sites/{site_id}/drive/root:/{parent_path}:/children"
                    
                    create_data = {
                        "name": part,
                        "folder": {},
                        "@microsoft.graph.conflictBehavior": "fail"
                    }
                    
                    create_response = requests.post(
                        create_url,
                        headers=self._get_headers(),
                        json=create_data,
                        timeout=30
                    )
                    
                    if create_response.status_code not in [200, 201, 409]:
                        logger.warning(f"Failed to create folder {part}: {create_response.text}")
                        return False
                    
                    logger.info(f"Created folder: {current_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error ensuring folder exists: {e}")
            return False
    
    def upload_file(
        self,
        filename: str,
        content: bytes,
        folder_path: str = "UseCase-1",
        content_type: str = None
    ) -> Dict[str, Any]:
        """
        Upload a file to SharePoint.
        
        Args:
            filename: Name of the file (e.g., "report.docx" or "document.pdf")
            content: File content as bytes
            folder_path: Folder path within the document library
            content_type: Optional content type override
            
        Returns:
            Dict with success status and file details
        """
        try:
            # Ensure folder exists
            if not self.ensure_folder_exists(folder_path):
                return {"success": False, "error": "Failed to create folder"}
            
            site_id = self.get_site_id()
            
            # Determine content type
            if content_type is None:
                if filename.endswith(".docx"):
                    content_type = CONTENT_TYPE_DOCX
                elif filename.endswith(".pdf"):
                    content_type = CONTENT_TYPE_PDF
                else:
                    content_type = "application/octet-stream"
            
            # Upload file
            upload_url = f"{self.graph_base_url}/sites/{site_id}/drive/root:/{folder_path}/{filename}:/content"
            
            response = requests.put(
                upload_url,
                headers=self._get_headers(content_type),
                data=content,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"File uploaded: {result.get('webUrl', 'N/A')}")
            
            return {
                "success": True,
                "file_id": result.get("id"),
                "item_id": result.get("id"),  # Alias for metadata update
                "web_url": result.get("webUrl"),
                "name": result.get("name"),
                "size": result.get("size"),
                "created_datetime": result.get("createdDateTime")
            }
            
        except requests.exceptions.HTTPError as e:
            error_detail = e.response.text if e.response else str(e)
            logger.error(f"SharePoint upload failed: {error_detail}")
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {error_detail}" if e.response else str(e)
            }
        except Exception as e:
            logger.error(f"SharePoint upload error: {e}")
            return {"success": False, "error": str(e)}
    
    def update_file_metadata(
        self,
        item_id: str,
        category: str
    ) -> Dict[str, Any]:
        """
        Update the Category metadata column for a file.
        
        Args:
            item_id: The SharePoint item ID (returned from upload_file)
            category: The category value to set
            
        Returns:
            Dict with success status
        """
        try:
            site_id = self.get_site_id()
            
            # Update list item fields
            # The column internal name is typically "Category" but may vary
            update_url = f"{self.graph_base_url}/sites/{site_id}/drive/items/{item_id}/listItem/fields"
            
            update_data = {
                "Category": category
            }
            
            response = requests.patch(
                update_url,
                headers=self._get_headers(),
                json=update_data,
                timeout=30
            )
            response.raise_for_status()
            
            logger.info(f"Metadata updated for item {item_id}: Category={category}")
            return {"success": True, "category": category}
            
        except requests.exceptions.HTTPError as e:
            error_detail = e.response.text if e.response else str(e)
            logger.error(f"Metadata update failed: {error_detail}")
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {error_detail}" if e.response else str(e)
            }
        except Exception as e:
            logger.error(f"Metadata update error: {e}")
            return {"success": False, "error": str(e)}
    
    def upload_file_with_category(
        self,
        filename: str,
        content: bytes,
        folder_path: str,
        category: str
    ) -> Dict[str, Any]:
        """
        Upload a file and set its Category metadata in one call.
        
        Args:
            filename: Name of the file
            content: File content as bytes
            folder_path: Folder path within the document library
            category: The category value to set
            
        Returns:
            Dict with success status and file details
        """
        # First upload the file
        upload_result = self.upload_file(filename, content, folder_path)
        
        if not upload_result.get("success"):
            return upload_result
        
        # Then update metadata
        item_id = upload_result.get("item_id")
        if item_id and category:
            metadata_result = self.update_file_metadata(item_id, category)
            upload_result["metadata_update"] = metadata_result
        
        return upload_result
    
    def upload_extracted_data_as_json(
        self,
        extracted_data: Dict[str, Any],
        base_filename: str,
        document_type: str,
        sharepoint_folder: str = "UseCase-2"
    ) -> Dict[str, Any]:
        """
        Upload extracted data as a JSON file to SharePoint.
        
        Args:
            extracted_data: The extracted data dictionary
            base_filename: The base filename to use for the JSON file
            document_type: The type of the document
            sharepoint_folder: Folder path within the document library 
        """
        import json
        
        try:
            json_content = json.dumps(extracted_data, indent=2).encode("utf-8")
            json_filename = f"{base_filename}_extracted_data.json"
            
            upload_result = self.upload_file_with_category(
                filename=json_filename,
                content=json_content,
                folder_path=sharepoint_folder,
                category=document_type
            )
            
            return upload_result
            
        except Exception as e:
            logger.error(f"Error uploading extracted data as JSON: {e}")
            return {"success": False, "error": str(e)}

    def upload_extracted_data_as_excel(
        self,
        extracted_data: Dict[str, Any],
        base_filename: str,
        document_type: str,
        sharepoint_folder: str = "UseCase-2"
    ) -> Dict[str, Any]:
        """Upload extracted data as an Excel file to SharePoint."""
        try:
            import pandas as pd
            from io import BytesIO
            
            # Flatten the nested structure
            df = pd.json_normalize(extracted_data)
            
            # Create Excel file in memory
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Extracted Data')
            
            excel_content = excel_buffer.getvalue()
            excel_filename = f"{base_filename}_data.xlsx"
            
            upload_result = self.upload_file_with_category(
                filename=excel_filename,
                content=excel_content,
                folder_path=sharepoint_folder,
                category=document_type
            )
            
            return upload_result
            
        except Exception as e:
            logger.error(f"Error uploading extracted data as Excel: {e}")
            return {"success": False, "error": str(e)}

# Singleton instance
_sharepoint_service: Optional[SharePointService] = None


def get_sharepoint_service() -> SharePointService:
    """Get or create the SharePoint service singleton."""
    global _sharepoint_service
    if _sharepoint_service is None:
        _sharepoint_service = SharePointService()
    return _sharepoint_service