-- =============================================================================
-- 사용자 ID / 비밀번호 로그인용 스키마 (Supabase Auth 미사용)
-- =============================================================================
-- 선행 조건: `prompt/10.multi-session/multi-session-ref.sql` 로 rag_* 테이블이
--            이미 만들어져 있어야 합니다.
--
-- 실행: Supabase Dashboard → SQL Editor → 아래 전체 실행
--
-- 앱 설정: Streamlit/.env 에 SUPABASE_SERVICE_ROLE_KEY 권장 (app_users 접근)
-- =============================================================================

create extension if not exists vector;

create table if not exists public.app_users (
  username text primary key
    check (char_length(username) >= 1 and char_length(username) <= 128),
  password_hash text not null,
  salt text not null,
  created_at timestamptz not null default now()
);

comment on table public.app_users is '앱 자체 로그인 계정 (username + PBKDF2 해시). Supabase Auth 미사용.';

-- rag_sessions / rag_documents 에 앱 사용자 ID(문자열) 저장
alter table public.rag_sessions drop constraint if exists rag_sessions_user_id_fkey;
alter table public.rag_documents drop constraint if exists rag_documents_user_id_fkey;

alter table public.rag_sessions add column if not exists user_id text;
alter table public.rag_documents add column if not exists user_id text;

do $$
begin
  if exists (
    select 1 from information_schema.columns
    where table_schema = 'public' and table_name = 'rag_sessions'
      and column_name = 'user_id' and data_type = 'uuid'
  ) then
    alter table public.rag_sessions alter column user_id type text using user_id::text;
  end if;
  if exists (
    select 1 from information_schema.columns
    where table_schema = 'public' and table_name = 'rag_documents'
      and column_name = 'user_id' and data_type = 'uuid'
  ) then
    alter table public.rag_documents alter column user_id type text using user_id::text;
  end if;
end $$;

-- 사용자별로 동일 본문 해시 허용 (전역 unique 해제)
alter table public.rag_documents drop constraint if exists rag_documents_content_hash_key;

drop index if exists idx_rag_documents_user_content_hash;
create unique index if not exists idx_rag_documents_user_content_hash
  on public.rag_documents (user_id, content_hash);

create index if not exists idx_rag_sessions_user_updated
  on public.rag_sessions (user_id, updated_at desc);

-- 검색 RPC: 세션 소유자(app user id)와 일치할 때만 매칭
drop function if exists public.match_rag_documents(vector, uuid, integer, double precision);
drop function if exists public.match_rag_documents(vector, uuid, text, integer, double precision);

create or replace function public.match_rag_documents(
  query_embedding vector(1536),
  target_session_id uuid,
  match_app_user_id text,
  match_count integer default 5,
  match_threshold double precision default 0.6
)
returns table (
  document_id bigint,
  content text,
  metadata jsonb,
  similarity double precision
)
language sql
stable
security invoker
set search_path = public
as $$
  select
    d.id as document_id,
    d.content,
    d.metadata,
    1 - (d.embedding <=> query_embedding) as similarity
  from public.rag_documents d
  join public.rag_session_documents rsd on rsd.document_id = d.id
  join public.rag_sessions s on s.id = rsd.session_id
  where rsd.session_id = target_session_id
    and s.user_id is not distinct from match_app_user_id
    and (1 - (d.embedding <=> query_embedding)) >= match_threshold
  order by d.embedding <=> query_embedding
  limit match_count;
$$;

grant execute on function public.match_rag_documents(vector, uuid, text, integer, double precision)
  to anon, authenticated, service_role;

-- app_users RLS
-- - service_role: 전체 (운영 시 권장)
-- - anon/authenticated: 가입(INSERT) + 로그인(SELECT 해시) — service_role 없이 Streamlit 데모용.
--   보안: anon 키로 테이블을 직접 조회하면 해시가 노출될 수 있음.
alter table public.app_users enable row level security;

grant select, insert on public.app_users to anon, authenticated;

drop policy if exists "app_users_service_only" on public.app_users;
create policy "app_users_service_only" on public.app_users
  for all to service_role using (true) with check (true);

drop policy if exists "app_users_anon_insert" on public.app_users;
drop policy if exists "app_users_anon_select" on public.app_users;

create policy "app_users_anon_insert"
  on public.app_users
  for insert
  to anon, authenticated
  with check (true);

create policy "app_users_anon_select"
  on public.app_users
  for select
  to anon, authenticated
  using (true);
