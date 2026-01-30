#!/bin/bash

# NOTE, hyunnnchoi, 2026.01.30
# vllm-fetch 스크립트: fetch/rollback 모드 지원
# vllm-fetch 폴더의 파이썬 파일들을 각 파일 최상단에 명시된 경로로 자동 복사/복원
# Usage: ./fetch.sh [fetch|rollback]

set -e  # Exit on error

# 색상 정의
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 디렉토리 경로
FETCH_DIR="/home/work/hyunmokchoi/vllm-fetch/vllm-fetch"
BACKUP_DIR="$FETCH_DIR/.backup"
MANIFEST_FILE="$BACKUP_DIR/manifest.txt"

# 폴더 소유권 변경 함수
change_ownership() {
    echo -e "${BLUE}[1/3] 폴더 소유권 변경 중...${NC}"

    # /vllm 및 /lmcache 폴더의 소유권을 현재 사용자로 변경
    if [ -d "/vllm" ]; then
        echo -e "${YELLOW}  - /vllm 폴더 소유권 변경 (sudo 권한 필요)${NC}"
        if sudo chown -R $USER:$USER /vllm 2>/dev/null; then
            echo -e "${GREEN}    ✓ /vllm 폴더 소유권 변경 완료${NC}"
        else
            echo -e "${RED}    ✗ /vllm 폴더 소유권 변경 실패 (sudo 권한 필요 또는 이미 권한 있음)${NC}"
        fi
    else
        echo -e "${YELLOW}  - /vllm 폴더가 존재하지 않습니다${NC}"
    fi

    if [ -d "/lmcache" ]; then
        echo -e "${YELLOW}  - /lmcache 폴더 소유권 변경 (sudo 권한 필요)${NC}"
        if sudo chown -R $USER:$USER /lmcache 2>/dev/null; then
            echo -e "${GREEN}    ✓ /lmcache 폴더 소유권 변경 완료${NC}"
        else
            echo -e "${RED}    ✗ /lmcache 폴더 소유권 변경 실패 (sudo 권한 필요 또는 이미 권한 있음)${NC}"
        fi
    else
        echo -e "${YELLOW}  - /lmcache 폴더가 존재하지 않습니다${NC}"
    fi

    echo ""
}

# Fetch 모드 함수
fetch_files() {
    echo -e "${BLUE}=== Starting vllm-fetch mode ===${NC}"
    echo ""

    change_ownership

    echo -e "${BLUE}[2/3] 파일 백업 및 복사 중...${NC}"

    # 백업 디렉토리 생성
    mkdir -p "$BACKUP_DIR"

    # 백업 디렉토리가 root 소유인 경우 소유권 변경
    if [ -d "$BACKUP_DIR" ]; then
        sudo chown -R $USER:$USER "$BACKUP_DIR" 2>/dev/null || true
    fi

    # 기존 manifest 파일이 있으면 삭제 (sudo 권한으로 생성된 파일도 처리)
    if [ -f "$MANIFEST_FILE" ]; then
        rm "$MANIFEST_FILE" 2>/dev/null || sudo rm "$MANIFEST_FILE" 2>/dev/null || true
    fi

    # 복사된 파일 수 카운터
    COPIED_COUNT=0
    FAILED_COUNT=0

    # vllm-fetch 폴더의 모든 .py 파일에 대해 반복
    for source_file in "$FETCH_DIR"/*.py; do
        # 파일이 존재하는지 확인
        if [ ! -f "$source_file" ]; then
            continue
        fi

        # 파일명 추출
        filename=$(basename "$source_file")

        # 첫 번째 줄 읽기 (주석에서 경로 추출)
        first_line=$(head -n 1 "$source_file")

        # 주석에서 경로 추출 (# 와 공백 제거)
        destination=$(echo "$first_line" | sed 's/^#\s*//')

        # 경로가 비어있거나 주석이 아닌 경우 건너뛰기
        if [[ -z "$destination" ]] || [[ ! "$first_line" =~ ^#.* ]]; then
            echo -e "${YELLOW}  - $filename: 경로 정보가 없습니다. 건너뜁니다.${NC}"
            continue
        fi

        # 경로가 절대 경로인지 확인
        if [[ ! "$destination" =~ ^/ ]]; then
            echo -e "${RED}  - $filename: 상대 경로는 지원하지 않습니다 ($destination)${NC}"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            continue
        fi

        # 기존 파일이 있으면 백업
        if [ -f "$destination" ]; then
            # 경로를 파일명으로 변환 (/ -> _)
            backup_filename=$(echo "$destination" | sed 's/\//_/g' | sed 's/^_//')
            backup_path="$BACKUP_DIR/$backup_filename"

            echo -e "${YELLOW}  - 백업: $destination -> $backup_path${NC}"
            cp "$destination" "$backup_path"

            # manifest 파일에 기록 (복원 시 사용)
            echo "$destination|$backup_path" >> "$MANIFEST_FILE"
        else
            # 새 파일인 경우 manifest에 표시
            echo "$destination|NEW" >> "$MANIFEST_FILE"
        fi

        # 대상 디렉토리가 존재하는지 확인하고, 없으면 생성
        dest_dir=$(dirname "$destination")
        if [ ! -d "$dest_dir" ]; then
            echo -e "${YELLOW}  - 디렉토리 생성: $dest_dir${NC}"
            mkdir -p "$dest_dir"
        fi

        # 파일 복사
        echo -e "${GREEN}  - $filename -> $destination${NC}"
        cp "$source_file" "$destination"

        # 권한 설정 (읽기/쓰기 권한)
        chmod 644 "$destination"

        COPIED_COUNT=$((COPIED_COUNT + 1))
    done

    echo ""
    echo -e "${BLUE}[3/3] 완료${NC}"
    echo -e "${GREEN}복사 완료: ${COPIED_COUNT}개 파일${NC}"

    if [ $FAILED_COUNT -gt 0 ]; then
        echo -e "${RED}실패: ${FAILED_COUNT}개 파일${NC}"
    fi

    echo ""
    echo -e "${GREEN}✓ vllm-fetch 완료!${NC}"
    echo -e "${BLUE}백업 위치: $BACKUP_DIR${NC}"
}

# Rollback 모드 함수
rollback_files() {
    echo -e "${BLUE}=== Starting vllm-rollback mode ===${NC}"
    echo ""

    change_ownership

    echo -e "${BLUE}[2/3] 파일 복원 중...${NC}"

    # manifest 파일 확인
    if [ ! -f "$MANIFEST_FILE" ]; then
        echo -e "${RED}Error: Manifest 파일이 없습니다 ($MANIFEST_FILE)${NC}"
        echo -e "${YELLOW}백업이 없거나 fetch를 먼저 실행해야 합니다.${NC}"
        exit 1
    fi

    RESTORED_COUNT=0
    REMOVED_COUNT=0
    FAILED_COUNT=0

    # manifest 파일 읽어서 복원
    while IFS='|' read -r destination backup_path; do
        if [ "$backup_path" = "NEW" ]; then
            # 새로 생성된 파일이면 삭제
            if [ -f "$destination" ]; then
                echo -e "${YELLOW}  - 삭제: $destination (새로 생성된 파일)${NC}"
                rm "$destination"
                REMOVED_COUNT=$((REMOVED_COUNT + 1))
            fi
        else
            # 백업 파일 복원
            if [ -f "$backup_path" ]; then
                echo -e "${GREEN}  - 복원: $backup_path -> $destination${NC}"
                cp "$backup_path" "$destination"
                RESTORED_COUNT=$((RESTORED_COUNT + 1))
            else
                echo -e "${RED}  - 실패: 백업 파일을 찾을 수 없습니다 ($backup_path)${NC}"
                FAILED_COUNT=$((FAILED_COUNT + 1))
            fi
        fi
    done < "$MANIFEST_FILE"

    echo ""
    echo -e "${BLUE}[3/3] 완료${NC}"
    echo -e "${GREEN}복원 완료: ${RESTORED_COUNT}개 파일${NC}"

    if [ $REMOVED_COUNT -gt 0 ]; then
        echo -e "${YELLOW}삭제 완료: ${REMOVED_COUNT}개 파일 (새로 생성된 파일)${NC}"
    fi

    if [ $FAILED_COUNT -gt 0 ]; then
        echo -e "${RED}실패: ${FAILED_COUNT}개 파일${NC}"
    fi

    echo ""
    echo -e "${GREEN}✓ vllm-rollback 완료!${NC}"
}

# Main logic
MODE="${1:-fetch}"

case "$MODE" in
    fetch)
        fetch_files
        ;;
    rollback)
        rollback_files
        ;;
    *)
        echo "Usage: $0 [fetch|rollback]"
        echo "  fetch    - 파일 백업 및 복사 (default)"
        echo "  rollback - 백업된 파일 복원"
        exit 1
        ;;
esac
